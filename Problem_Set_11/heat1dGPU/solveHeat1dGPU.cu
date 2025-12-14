//
// solve the heat equation in 1D in parallel using CUDA
//
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "heat1d_GPU.h"

// Let's define a new type "Real" number which is equivalent to double
typedef double Real;

// Macro to handle error
# define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__))
static void HandleError( cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) 
    {
        printf("%s in %s at line %d \n", cudaGetErrorString( err ), file, line );
        exit ( EXIT_FAILURE );
    }
}

// Constant device memory for parameters
static const int numIntPar = IPAR_SIZE, numRealPar = RPAR_SIZE;
__device__ static int ipar_d[numIntPar]; // integer parameters
__device__ static Real rpar_d[numRealPar]; // real parameters

// Device functions to compute manufactured solutions
__device__ Real UTRUE_device(Real x, Real t)
{
    int &solution = ipar_d[IPAR_SOLUTION];
    Real &kxPi = rpar_d[RPAR_KXPI];
    Real &kappaPiSq = rpar_d[RPAR_KAPPOPISQ];
    
    if (solution == TRIG_DD) {
        return sin(kxPi * x) * exp(-kappaPiSq * t);
    }
    else if (solution == TRIG_NN) {
        return cos(kxPi * x) * exp(-kappaPiSq * t);
    }
    else if (solution == POLY_DD || solution == POLY_NN) {
        Real &b0 = rpar_d[RPAR_B0];
        Real &b1 = rpar_d[RPAR_B1];
        Real &b2 = rpar_d[RPAR_B2];
        Real &a0 = rpar_d[RPAR_A0];
        Real &a1 = rpar_d[RPAR_A1];
        return (b0 + x * (b1 + x * b2)) * (a0 + t * a1);
    }
    return 0.0;
}

__device__ Real UTRUEX_device(Real x, Real t)
{
    int &solution = ipar_d[IPAR_SOLUTION];
    Real &kxPi = rpar_d[RPAR_KXPI];
    Real &kappaPiSq = rpar_d[RPAR_KAPPOPISQ];
    
    if (solution == TRIG_DD) {
        return kxPi * cos(kxPi * x) * exp(-kappaPiSq * t);
    }
    else if (solution == TRIG_NN) {
        return -kxPi * sin(kxPi * x) * exp(-kappaPiSq * t);
    }
    else if (solution == POLY_DD || solution == POLY_NN) {
        Real &b1 = rpar_d[RPAR_B1];
        Real &b2 = rpar_d[RPAR_B2];
        Real &a0 = rpar_d[RPAR_A0];
        Real &a1 = rpar_d[RPAR_A1];
        return (b1 + 2.0 * x * b2) * (a0 + t * a1);
    }
    return 0.0;
}

__device__ Real FORCE_device(Real x, Real t)
{
    int &solution = ipar_d[IPAR_SOLUTION];
    
    if (solution == TRIG_DD || solution == TRIG_NN) {
        return 0.0;
    }
    else if (solution == POLY_DD || solution == POLY_NN) {
        Real &b0 = rpar_d[RPAR_B0];
        Real &b1 = rpar_d[RPAR_B1];
        Real &b2 = rpar_d[RPAR_B2];
        Real &a0 = rpar_d[RPAR_A0];
        Real &a1 = rpar_d[RPAR_A1];
        Real &kappa = rpar_d[RPAR_KAPPA];
        
        // UTRUET
        Real utruet = (b0 + x * (b1 + x * b2)) * a1;
        // UTRUEXX
        Real utruexx = 2.0 * b2 * (a0 + t * a1);
        // force = u_t - kappa*u_xx
        return utruet - kappa * utruexx;
    }
    return 0.0;
}

// CUDA kernel to take heat step
__global__ void takeHeatStepGPU(Real *uc, Real *un, Real *x, Real t)
{
    // Get parameters from constant memory using references
    int &n1a = ipar_d[IPAR_N1A];
    int &n1b = ipar_d[IPAR_N1B];
    int &nd1a = ipar_d[IPAR_ND1A];
    
    Real &rx = rpar_d[RPAR_RX];
    Real &dt = rpar_d[RPAR_DT];
    
    // Compute global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x + n1a;
    
    // Update interior points only
    if (i >= n1a && i <= n1b) {
        int idx = i - nd1a;
        Real force = FORCE_device(x[idx], t);
        un[idx] = uc[idx] + rx * (uc[idx + 1] - 2.0 * uc[idx] + uc[idx - 1]) + dt * force;
    }
}

// CUDA kernel to apply left boundary condition
__global__ void applyBoundaryConditionLeft(Real *u, Real *x, Real t)
{
    int &n1a = ipar_d[IPAR_N1A];
    int &nd1a = ipar_d[IPAR_ND1A];
    int &bcLeft = ipar_d[IPAR_BC_LEFT];
    Real &dx = rpar_d[RPAR_DX];
    
    int i = n1a;
    int idx = i - nd1a;
    int is = 1; // is = 1 on left side
    
    if (bcLeft == BC_DIRICHLET) {
        // Set boundary value
        u[idx] = UTRUE_device(x[idx], t);
        // Extrapolate ghost cell
        u[idx - is] = 3.0 * u[idx] - 3.0 * u[idx + is] + u[idx + 2*is];
    }
    else if (bcLeft == BC_NEUMANN) {
        // Set ghost cell using derivative
        u[idx - is] = u[idx + is] - 2.0 * is * dx * UTRUEX_device(x[idx], t);
    }
}

// CUDA kernel to apply right boundary condition
__global__ void applyBoundaryConditionRight(Real *u, Real *x, Real t)
{
    int &n1b = ipar_d[IPAR_N1B];
    int &nd1a = ipar_d[IPAR_ND1A];
    int &bcRight = ipar_d[IPAR_BC_RIGHT];
    Real &dx = rpar_d[RPAR_DX];
    
    int i = n1b;
    int idx = i - nd1a;
    int is = -1; // is = -1 on right side
    
    if (bcRight == BC_DIRICHLET) {
        // Set boundary value
        u[idx] = UTRUE_device(x[idx], t);
        // Extrapolate ghost cell
        u[idx - is] = 3.0 * u[idx] - 3.0 * u[idx + is] + u[idx + 2*is];
    }
    else if (bcRight == BC_NEUMANN) {
        // Set ghost cell using derivative
        u[idx - is] = u[idx + is] - 2.0 * is * dx * UTRUEX_device(x[idx], t);
    }
}

// Main GPU solver function
Real SolveHeat1dGPU(Real *u_initial, int numSteps, Real tFinal, int *ipar_h, 
                    Real *rpar_h, Real *x_h, Real &cpuTime, int &numThreads, int debug)
{
    // Get parameters from host arrays
    const int Nx = ipar_h[IPAR_NX];
    const int nd1 = ipar_h[IPAR_ND1];
    const int n1a = ipar_h[IPAR_N1A];
    const int n1b = ipar_h[IPAR_N1B];
    const int nd1a = ipar_h[IPAR_ND1A];
    const int nd1b = ipar_h[IPAR_ND1B];
    const int solution = ipar_h[IPAR_SOLUTION];
    
    const Real dt = rpar_h[RPAR_DT];
    const Real kxPi = rpar_h[RPAR_KXPI];
    const Real kappaPiSq = rpar_h[RPAR_KAPPOPISQ];

    const int bcLeft = ipar_h[IPAR_BC_LEFT];
    const int bcRight = ipar_h[IPAR_BC_RIGHT];
    const Real kappa = rpar_h[RPAR_KAPPA];
    const int numghost = 1;
    
    // Determine solution name
    const char *solutionName = "";
    if (solution == TRIG_DD) solutionName = "trigDD";
    else if (solution == TRIG_NN) solutionName = "trigNN";
    else if (solution == POLY_DD) solutionName = "polyDD";
    else if (solution == POLY_NN) solutionName = "polyNN";
    
    // Configure kernel launch parameters
    const int Nt = 128;
    numThreads = Nt;
    const int Nb = (int)ceil((1.0*Nx)/Nt);
  
    printf("------------------- GPU: Solve the heat equation in 1D solution=%s --------------------\n", solutionName);
    printf("Nb=%d, Nt=%d\n", Nb, Nt);
    printf("numGhost=%d, n1a=%d, n1b=%d, nd1a=%d, nd1b=%d, debug=%d\n",numghost, n1a, n1b, nd1a, nd1b, debug);
    printf("numSteps=%d, nx=%d, kappa=%g, tFinal=%g, boundaryCondition(0,0)=%d, boundaryCondition(1,0)=%d\n",
           numSteps, Nx, kappa, tFinal, bcLeft, bcRight);

    // To compute error on CPU we compute the exact solution in CPU 
    auto UTRUE_host = [&](Real x, Real t) -> Real {
        if (solution == TRIG_DD) {
            return sin(kxPi * x) * exp(-kappaPiSq * t);
        }
        else if (solution == TRIG_NN) {
            return cos(kxPi * x) * exp(-kappaPiSq * t);
        }
        else if (solution == POLY_DD || solution == POLY_NN) {
            const Real b0 = rpar_h[RPAR_B0];
            const Real b1 = rpar_h[RPAR_B1];
            const Real b2 = rpar_h[RPAR_B2];
            const Real a0 = rpar_h[RPAR_A0];
            const Real a1 = rpar_h[RPAR_A1];
            return (b0 + x * (b1 + x * b2)) * (a0 + t * a1);
        }
        return 0.0;
    };

        // Allocate host array for debug checking (only if needed)
    Real *u_debug = nullptr;
    if (debug > 0) {
        u_debug = new Real[nd1];
        printf("  Debug mode enabled: will check error in every step\n");
    }
    
    // Copy parameters to constant device memory
    HANDLE_ERROR( cudaMemcpyToSymbol(ipar_d, ipar_h, numIntPar * sizeof(int)) );
    HANDLE_ERROR( cudaMemcpyToSymbol(rpar_d, rpar_h, numRealPar * sizeof(Real)) );
    
    // Allocate device memory for solution arrays and grid
    Real *u_d[2], *x_d;
    HANDLE_ERROR( cudaMalloc((void**)&u_d[0], nd1 * sizeof(Real)) );
    HANDLE_ERROR( cudaMalloc((void**)&u_d[1], nd1 * sizeof(Real)) );
    HANDLE_ERROR( cudaMalloc((void**)&x_d, nd1 * sizeof(Real)) );
    
    // Copy initial condition and grid to device
    HANDLE_ERROR( cudaMemcpy(u_d[0], u_initial, nd1 * sizeof(Real), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(x_d, x_h, nd1 * sizeof(Real), cudaMemcpyHostToDevice) );
    
    // Start timing
    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start) );
    
    // Time-stepping loop - ALL ON GPU
    for (int n = 0; n < numSteps; n++) {
        Real t = n * dt;
        
        int cur = n % 2;
        int next = (n + 1) % 2;
        
        // Update interior points on GPU
        takeHeatStepGPU<<<Nb, Nt>>>(u_d[cur], u_d[next], x_d, t);
        HANDLE_ERROR( cudaGetLastError() );
        
        // Synchronize before applying boundary conditions
        HANDLE_ERROR( cudaDeviceSynchronize() );
        
        // Apply boundary conditions on GPU (small kernels with 1 block, 1 thread)
        Real t_next = t + dt;
        applyBoundaryConditionLeft<<<1, 1>>>(u_d[next], x_d, t_next);
        HANDLE_ERROR( cudaGetLastError() );
        
        applyBoundaryConditionRight<<<1, 1>>>(u_d[next], x_d, t_next);
        HANDLE_ERROR( cudaGetLastError() );
        
        // Synchronize after boundary conditions
        HANDLE_ERROR( cudaDeviceSynchronize() );

        // If debug enabled copies eash step file and print in CPU
        if (debug > 0) {
            // Copy current solution from GPU to CPU
            HANDLE_ERROR( cudaMemcpy(u_debug, u_d[next], nd1 * sizeof(Real), 
                                    cudaMemcpyDeviceToHost) );
            
            // Compute error against exact solution
            Real maxErr = 0.0;
            for (int i = nd1a; i <= nd1b; i++) {
                int idx = i - nd1a;
                Real uExact = UTRUE_host(x_h[idx], t_next);
                Real err = fabs(u_debug[idx] - uExact);
                maxErr = fmax(maxErr, err);
            }
            
            printf("  GPU step=%d, t=%9.3e, maxErr=%9.2e\n", n+1, t_next, maxErr);
        }
    }
    
    // Stop timing
    HANDLE_ERROR( cudaEventRecord(stop) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    
    float time_ms = 0;
    HANDLE_ERROR( cudaEventElapsedTime(&time_ms, start, stop) );
    cpuTime = time_ms / 1000.0; // Convert to seconds
    
    // Copy final solution back to host for error computation
    int finalIndex = numSteps % 2;
    Real *u_final = new Real[nd1];
    HANDLE_ERROR( cudaMemcpy(u_final, u_d[finalIndex], nd1 * sizeof(Real), cudaMemcpyDeviceToHost) );
    
    Real maxErr = 0.0;
    for (int i = nd1a; i <= nd1b; i++) {
        int idx = i - nd1a;
        Real err = fabs(u_final[idx] - UTRUE_host(x_h[idx], tFinal));
        maxErr = fmax(maxErr, err);
    }
    
    printf("GPU: numSteps=%d, nx=%d, maxErr=%8.2e, cpu=%8.2e(s)\n",
           numSteps, Nx, maxErr, cpuTime);
    
    HANDLE_ERROR( cudaFree(u_d[0]) );
    HANDLE_ERROR( cudaFree(u_d[1]) );
    HANDLE_ERROR( cudaFree(x_d) );
    HANDLE_ERROR( cudaEventDestroy(start) );
    HANDLE_ERROR( cudaEventDestroy(stop) );
    delete[] u_final;
    if (u_debug != nullptr) {
        delete[] u_debug;
    }
    
    return maxErr;
}