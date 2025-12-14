//
// solve the heat equation in 2D using CUDA
//
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "heat2d_GPU.h"

// Let's define a new type "Real" number which is equivalent to double
typedef double Real;

// Macro to handle error
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) 
    {
        printf("%s in %s at line %d \n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// Constant device memory for parameters
static const int numIntPar = IPAR_SIZE, numRealPar = RPAR_SIZE;
__device__ static int ipar_d[numIntPar];     // integer parameters
__device__ static Real rpar_d[numRealPar];   // real parameters

// Device functions to compute manufactured solutions
__device__ Real UTRUE_device(Real x, Real y, Real t)
{
    int &solution = ipar_d[IPAR_SOLUTION];
    
    if (solution == TRIG_DD) {
        Real &kxPi = rpar_d[RPAR_KXPI];
        Real &kyPi = rpar_d[RPAR_KYPI];
        Real &kappaK2 = rpar_d[RPAR_KAPPA_K2];
        return sin(kxPi * x) * sin(kyPi * y) * exp(-kappaK2 * t);
    }
    else if (solution == MANUFACTURED) {
        Real &b0 = rpar_d[RPAR_B0];
        Real &b1 = rpar_d[RPAR_B1];
        Real &b2 = rpar_d[RPAR_B2];
        Real &c0 = rpar_d[RPAR_C0];
        Real &c1 = rpar_d[RPAR_C1];
        Real &c2 = rpar_d[RPAR_C2];
        Real &a0 = rpar_d[RPAR_A0];
        Real &a1 = rpar_d[RPAR_A1];
        Real &a2 = rpar_d[RPAR_A2];
        return (b0 + x * (b1 + x * b2)) * (c0 + y * (c1 + y * c2)) * (a0 + t * (a1 + t * a2));
    }
    return 0.0;
}

__device__ Real FORCE_device(Real x, Real y, Real t)
{
    int &solution = ipar_d[IPAR_SOLUTION];
    
    if (solution == TRIG_DD) {
        return 0.0;
    }
    else if (solution == MANUFACTURED) {
        Real &b0 = rpar_d[RPAR_B0];
        Real &b1 = rpar_d[RPAR_B1];
        Real &b2 = rpar_d[RPAR_B2];
        Real &c0 = rpar_d[RPAR_C0];
        Real &c1 = rpar_d[RPAR_C1];
        Real &c2 = rpar_d[RPAR_C2];
        Real &a0 = rpar_d[RPAR_A0];
        Real &a1 = rpar_d[RPAR_A1];
        Real &a2 = rpar_d[RPAR_A2];
        Real &kappa = rpar_d[RPAR_KAPPA];
        
        // UTRUET 
        Real utruet = (b0 + x * (b1 + x * b2)) * (c0 + y * (c1 + y * c2)) * (a1 + 2.0 * t * a2);
        
        // UTRUEXX 
        Real utruexx = (2.0 * b2) * (c0 + y * (c1 + y * c2)) * (a0 + t * (a1 + t * a2));
        
        // UTRUEYY 
        Real utrueyy = (b0 + x * (b1 + x * b2)) * (2.0 * c2) * (a0 + t * (a1 + t * a2));
        
        // force 
        return utruet - kappa * (utruexx + utrueyy);
    }
    return 0.0;
}

// Macro for 1D thread indexing
#define INDEX1(tx,bx) ((tx) + blockDim.x * (bx))

// CUDA kernel to take heat step in 2D
__global__ void takeHeatStep2dGPU(Real *uc, Real *un, Real *x, Real *y, Real t)
{
    // Get parameters from constant memory
    int &n1a = ipar_d[IPAR_N1A];
    int &n1b = ipar_d[IPAR_N1B];
    int &n2a = ipar_d[IPAR_N2A];
    int &n2b = ipar_d[IPAR_N2B];
    int &nd1a = ipar_d[IPAR_ND1A];
    int &nd2a = ipar_d[IPAR_ND2A];
    int &nd1 = ipar_d[IPAR_ND1];
    
    Real &rx = rpar_d[RPAR_RX];
    Real &ry = rpar_d[RPAR_RY];
    Real &dt = rpar_d[RPAR_DT];
    
    // Compute 2D indices from 1D thread index
    int tid = INDEX1(threadIdx.x, blockIdx.x);
    int i2 = tid / nd1 + nd2a;
    int i1 = tid - nd1 * (i2 - nd2a);
    
    // Macros for 2D array indexing
    #define UC(i1,i2) uc[(i1-nd1a) + nd1*(i2-nd2a)]
    #define UN(i1,i2) un[(i1-nd1a) + nd1*(i2-nd2a)]
    #define X(i1,i2) x[(i1-nd1a) + nd1*(i2-nd2a)]
    #define Y(i1,i2) y[(i1-nd1a) + nd1*(i2-nd2a)]
    
    // Update interior points only
    if (i1 >= n1a && i1 <= n1b && i2 >= n2a && i2 <= n2b) 
    {
        Real force = FORCE_device(X(i1,i2), Y(i1,i2), t);
        UN(i1,i2) = UC(i1,i2) + rx * (UC(i1+1,i2) - 2.0 * UC(i1,i2) + UC(i1-1,i2))
                              + ry * (UC(i1,i2+1) - 2.0 * UC(i1,i2) + UC(i1,i2-1))
                              + dt * force;
    }
    
    #undef UC
    #undef UN
    #undef X
    #undef Y
}

// CUDA kernel to apply left boundary condition (side=0, axis=0)
__global__ void applyBoundaryConditionLeft(Real *u, Real *x, Real *y, Real t)
{
    int &n1a = ipar_d[IPAR_N1A];
    int &nd1a = ipar_d[IPAR_ND1A];
    int &nd2a = ipar_d[IPAR_ND2A];
    int &nd2b = ipar_d[IPAR_ND2B];
    int &nd1 = ipar_d[IPAR_ND1];
    int &bcLeft = ipar_d[IPAR_BC_LEFT];
    
    #define U(i1,i2) u[(i1-nd1a) + nd1*(i2-nd2a)]
    #define X(i1,i2) x[(i1-nd1a) + nd1*(i2-nd2a)]
    #define Y(i1,i2) y[(i1-nd1a) + nd1*(i2-nd2a)]
    
    int i2 = blockIdx.x * blockDim.x + threadIdx.x + nd2a;
    
    if (i2 >= nd2a && i2 <= nd2b) {
        int i1 = n1a;
        int is = 1; // is = 1 on left side
        
        if (bcLeft == BC_DIRICHLET) {
            U(i1,i2) = UTRUE_device(X(i1,i2), Y(i1,i2), t);
            U(i1-is,i2) = 3.0 * U(i1,i2) - 3.0 * U(i1+is,i2) + U(i1+2*is,i2);
        }
    }
    
    #undef U
    #undef X
    #undef Y
}

// CUDA kernel to apply right boundary condition (side=1, axis=0)
__global__ void applyBoundaryConditionRight(Real *u, Real *x, Real *y, Real t)
{
    int &n1b = ipar_d[IPAR_N1B];
    int &nd1a = ipar_d[IPAR_ND1A];
    int &nd2a = ipar_d[IPAR_ND2A];
    int &nd2b = ipar_d[IPAR_ND2B];
    int &nd1 = ipar_d[IPAR_ND1];
    int &bcRight = ipar_d[IPAR_BC_RIGHT];
    
    #define U(i1,i2) u[(i1-nd1a) + nd1*(i2-nd2a)]
    #define X(i1,i2) x[(i1-nd1a) + nd1*(i2-nd2a)]
    #define Y(i1,i2) y[(i1-nd1a) + nd1*(i2-nd2a)]
    
    int i2 = blockIdx.x * blockDim.x + threadIdx.x + nd2a;
    
    if (i2 >= nd2a && i2 <= nd2b) {
        int i1 = n1b;
        int is = -1; // is = -1 on right side
        
        if (bcRight == BC_DIRICHLET) {
            U(i1,i2) = UTRUE_device(X(i1,i2), Y(i1,i2), t);
            U(i1-is,i2) = 3.0 * U(i1,i2) - 3.0 * U(i1+is,i2) + U(i1+2*is,i2);
        }
    }
    
    #undef U
    #undef X
    #undef Y
}

// CUDA kernel to apply bottom boundary condition (side=0, axis=1)
__global__ void applyBoundaryConditionBottom(Real *u, Real *x, Real *y, Real t)
{
    int &n2a = ipar_d[IPAR_N2A];
    int &nd1a = ipar_d[IPAR_ND1A];
    int &nd1b = ipar_d[IPAR_ND1B];
    int &nd2a = ipar_d[IPAR_ND2A];
    int &nd1 = ipar_d[IPAR_ND1];
    int &bcBottom = ipar_d[IPAR_BC_BOTTOM];
    
    #define U(i1,i2) u[(i1-nd1a) + nd1*(i2-nd2a)]
    #define X(i1,i2) x[(i1-nd1a) + nd1*(i2-nd2a)]
    #define Y(i1,i2) y[(i1-nd1a) + nd1*(i2-nd2a)]
    
    int i1 = blockIdx.x * blockDim.x + threadIdx.x + nd1a;
    
    if (i1 >= nd1a && i1 <= nd1b) {
        int i2 = n2a;
        int is = 1; // is = 1 on bottom
        
        if (bcBottom == BC_DIRICHLET) {
            U(i1,i2) = UTRUE_device(X(i1,i2), Y(i1,i2), t);
            U(i1,i2-is) = 3.0 * U(i1,i2) - 3.0 * U(i1,i2+is) + U(i1,i2+2*is);
        }
    }
    
    #undef U
    #undef X
    #undef Y
}

// CUDA kernel to apply top boundary condition (side=1, axis=1)
__global__ void applyBoundaryConditionTop(Real *u, Real *x, Real *y, Real t)
{
    int &n2b = ipar_d[IPAR_N2B];
    int &nd1a = ipar_d[IPAR_ND1A];
    int &nd1b = ipar_d[IPAR_ND1B];
    int &nd2a = ipar_d[IPAR_ND2A];
    int &nd1 = ipar_d[IPAR_ND1];
    int &bcTop = ipar_d[IPAR_BC_TOP];
    
    #define U(i1,i2) u[(i1-nd1a) + nd1*(i2-nd2a)]
    #define X(i1,i2) x[(i1-nd1a) + nd1*(i2-nd2a)]
    #define Y(i1,i2) y[(i1-nd1a) + nd1*(i2-nd2a)]
    
    int i1 = blockIdx.x * blockDim.x + threadIdx.x + nd1a;
    
    if (i1 >= nd1a && i1 <= nd1b) {
        int i2 = n2b;
        int is = -1; // is = -1 on top
        
        if (bcTop == BC_DIRICHLET) {
            U(i1,i2) = UTRUE_device(X(i1,i2), Y(i1,i2), t);
            U(i1,i2-is) = 3.0 * U(i1,i2) - 3.0 * U(i1,i2+is) + U(i1,i2+2*is);
        }
    }
    
    #undef U
    #undef X
    #undef Y
}

// Main GPU solver function
Real SolveHeat2dGPU(Real *u_initial, int numSteps, Real tFinal, 
                    int *ipar_h, Real *rpar_h, Real *x_h, Real *y_h,
                    Real &cpuTime, int &numThreads, int debug)
{
    // Get parameters from host arrays
    const int nx = ipar_h[IPAR_NX];
    const int ny = ipar_h[IPAR_NY];
    const int nd1 = ipar_h[IPAR_ND1];
    const int nd2 = ipar_h[IPAR_ND2];
    const int n1a = ipar_h[IPAR_N1A];
    const int n1b = ipar_h[IPAR_N1B];
    const int n2a = ipar_h[IPAR_N2A];
    const int n2b = ipar_h[IPAR_N2B];
    const int nd1a = ipar_h[IPAR_ND1A];
    const int nd1b = ipar_h[IPAR_ND1B];
    const int nd2a = ipar_h[IPAR_ND2A];
    const int nd2b = ipar_h[IPAR_ND2B];
    const int solution = ipar_h[IPAR_SOLUTION];
    
    const Real dt = rpar_h[RPAR_DT];
    const Real kappa = rpar_h[RPAR_KAPPA];
    
    // Determine solution name
    const char *solutionName = "";
    if (solution == TRIG_DD) solutionName = "trueSolution";
    else if (solution == MANUFACTURED) solutionName = "manufacturedSolution";
    
    // Configure kernel launch parameters
    const int totalPoints = nd1 * nd2;
    const int Nt = 128;
    numThreads = Nt;
    const int Nb = (int)ceil((1.0 * totalPoints) / Nt);
    
    // For boundary conditions
    const int Nb1 = (int)ceil((1.0 * nd1) / Nt);
    const int Nb2 = (int)ceil((1.0 * nd2) / Nt);
    
    printf("------------------- GPU: Solve the heat equation in 2D solution=%s --------------------\n", solutionName);
    printf("Nb=%d, Nt=%d (for interior updates)\n", Nb, Nt);
    printf("Nb1=%d, Nb2=%d, Nt=%d (for boundary conditions)\n", Nb1, Nb2, Nt);
    printf("numSteps=%d, nx=%d, ny=%d, kappa=%g, tFinal=%g, debug=%d\n",
           numSteps, nx, ny, kappa, tFinal, debug);
    
    // Lambda for computing exact solution on CPU
    auto UTRUE_host = [&](Real x, Real y, Real t) -> Real {
        if (solution == TRIG_DD) {
            const Real kxPi = rpar_h[RPAR_KXPI];
            const Real kyPi = rpar_h[RPAR_KYPI];
            const Real kappaK2 = rpar_h[RPAR_KAPPA_K2];
            return sin(kxPi * x) * sin(kyPi * y) * exp(-kappaK2 * t);
        }
        else if (solution == MANUFACTURED) {
            const Real b0 = rpar_h[RPAR_B0];
            const Real b1 = rpar_h[RPAR_B1];
            const Real b2 = rpar_h[RPAR_B2];
            const Real c0 = rpar_h[RPAR_C0];
            const Real c1 = rpar_h[RPAR_C1];
            const Real c2 = rpar_h[RPAR_C2];
            const Real a0 = rpar_h[RPAR_A0];
            const Real a1 = rpar_h[RPAR_A1];
            const Real a2 = rpar_h[RPAR_A2];
            return (b0 + x * (b1 + x * b2)) * (c0 + y * (c1 + y * c2)) * (a0 + t * (a1 + t * a2));
        }
        return 0.0;
    };
    
    // Allocate host array for debug checking (only if needed)
    Real *u_debug = nullptr;
    if (debug > 0) {
        u_debug = new Real[nd1 * nd2];
        printf("  Debug mode enabled: will check error in every step\n");
    }
    
    // Copy parameters to constant device memory
    HANDLE_ERROR(cudaMemcpyToSymbol(ipar_d, ipar_h, numIntPar * sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(rpar_d, rpar_h, numRealPar * sizeof(Real)));
    
    // Allocate device memory for solution arrays and grid
    Real *u_d[2], *x_d, *y_d;
    HANDLE_ERROR(cudaMalloc((void**)&u_d[0], nd1 * nd2 * sizeof(Real)));
    HANDLE_ERROR(cudaMalloc((void**)&u_d[1], nd1 * nd2 * sizeof(Real)));
    HANDLE_ERROR(cudaMalloc((void**)&x_d, nd1 * nd2 * sizeof(Real)));
    HANDLE_ERROR(cudaMalloc((void**)&y_d, nd1 * nd2 * sizeof(Real)));
    
    // Copy initial condition and grid to device
    HANDLE_ERROR(cudaMemcpy(u_d[0], u_initial, nd1 * nd2 * sizeof(Real), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(x_d, x_h, nd1 * nd2 * sizeof(Real), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(y_d, y_h, nd1 * nd2 * sizeof(Real), cudaMemcpyHostToDevice));
    
    // Start timing
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start));
    
    // Time-stepping loop - ALL ON GPU
    for (int n = 0; n < numSteps; n++) {
        Real t = n * dt;
        
        int cur = n % 2;
        int next = (n + 1) % 2;
        
        // Update interior points on GPU
        takeHeatStep2dGPU<<<Nb, Nt>>>(u_d[cur], u_d[next], x_d, y_d, t);
        HANDLE_ERROR(cudaGetLastError());
        
        // Synchronize before applying boundary conditions
        HANDLE_ERROR(cudaDeviceSynchronize());
        
        // Apply boundary conditions on GPU
        Real t_next = t + dt;
        applyBoundaryConditionLeft<<<Nb2, Nt>>>(u_d[next], x_d, y_d, t_next);
        HANDLE_ERROR(cudaGetLastError());
        
        applyBoundaryConditionRight<<<Nb2, Nt>>>(u_d[next], x_d, y_d, t_next);
        HANDLE_ERROR(cudaGetLastError());
        
        applyBoundaryConditionBottom<<<Nb1, Nt>>>(u_d[next], x_d, y_d, t_next);
        HANDLE_ERROR(cudaGetLastError());
        
        applyBoundaryConditionTop<<<Nb1, Nt>>>(u_d[next], x_d, y_d, t_next);
        HANDLE_ERROR(cudaGetLastError());
        
        // Synchronize after boundary conditions
        HANDLE_ERROR(cudaDeviceSynchronize());
        
        // If debug enabled, copy each step to CPU and print error
        if (debug > 0) {
            HANDLE_ERROR(cudaMemcpy(u_debug, u_d[next], nd1 * nd2 * sizeof(Real), 
                                    cudaMemcpyDeviceToHost));
            
            // Compute error against exact solution
            Real maxErr = 0.0;
            #define U_DEBUG(i1,i2) u_debug[(i1-nd1a) + nd1*(i2-nd2a)]
            #define X_H(i1,i2) x_h[(i1-nd1a) + nd1*(i2-nd2a)]
            #define Y_H(i1,i2) y_h[(i1-nd1a) + nd1*(i2-nd2a)]
            
            for (int i2 = n2a; i2 <= n2b; i2++) {
                for (int i1 = n1a; i1 <= n1b; i1++) {
                    Real uExact = UTRUE_host(X_H(i1,i2), Y_H(i1,i2), t_next);
                    Real err = fabs(U_DEBUG(i1,i2) - uExact);
                    maxErr = fmax(maxErr, err);
                }
            }
            
            printf("  GPU step=%d, t=%9.3e, maxErr=%9.2e\n", n+1, t_next, maxErr);
            
            #undef U_DEBUG
            #undef X_H
            #undef Y_H
        }
    }
    
    // Stop timing
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    
    float time_ms = 0;
    HANDLE_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
    cpuTime = time_ms / 1000.0; // Convert to seconds
    
    // Copy final solution back to host for error computation
    int finalIndex = numSteps % 2;
    Real *u_final = new Real[nd1 * nd2];
    HANDLE_ERROR(cudaMemcpy(u_final, u_d[finalIndex], nd1 * nd2 * sizeof(Real), cudaMemcpyDeviceToHost));
    
    // Compute final error
    Real maxErr = 0.0;
    #define U_FINAL(i1,i2) u_final[(i1-nd1a) + nd1*(i2-nd2a)]
    #define X_H(i1,i2) x_h[(i1-nd1a) + nd1*(i2-nd2a)]
    #define Y_H(i1,i2) y_h[(i1-nd1a) + nd1*(i2-nd2a)]
    
    for (int i2 = n2a; i2 <= n2b; i2++) {
        for (int i1 = n1a; i1 <= n1b; i1++) {
            Real err = fabs(U_FINAL(i1,i2) - UTRUE_host(X_H(i1,i2), Y_H(i1,i2), tFinal));
            maxErr = fmax(maxErr, err);
        }
    }
    
    printf("GPU: numSteps=%d, nx=%d, ny=%d, maxErr=%8.2e, cpu=%8.2e(s)\n",
           numSteps, nx, ny, maxErr, cpuTime);
    
    // Cleanup
    HANDLE_ERROR(cudaFree(u_d[0]));
    HANDLE_ERROR(cudaFree(u_d[1]));
    HANDLE_ERROR(cudaFree(x_d));
    HANDLE_ERROR(cudaFree(y_d));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    delete[] u_final;
    if (u_debug != nullptr) {
        delete[] u_debug;
    }
    
    #undef U_FINAL
    #undef X_H
    #undef Y_H
    
    return maxErr;
}