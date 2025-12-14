// GPU code to sum two vectors (of doubles) of length n 
// with benchmarking at different number of threads per block
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>

#define Real double

/*
void add (double *a, double *b , double *c, int *n)
adds the first two vectors and places the resut in the third vector
which are read from the cpu and added in GPU
*/

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

__global__ void add ( double *a_d, double *b_d, double *c_d, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = tid;
    if ( i < n)
    {   
        c_d[i] = a_d[i] + b_d[i];
    }
}

// Calculation of addition in CPU fro comparision
void addCPU (Real *a, Real *b, Real *c, int n)
{
    for (int i=0; i<n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// Compute the max norm 
Real computeMaxNorm ( Real *computed, Real *expected, int n)
{
    Real maxErr = 0.0;
    for (int i=0; i<n; i++)
    {
        Real error = fabs( computed[i] - expected[i]);
        if (error > maxErr)
        {
            maxErr = error;
        }
    }
    return maxErr;
}

int main (int argc, char* argv[])
{   
    if (argc != 2)
    {
        printf( " Usage: %s <vector_length>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]); // Get the length of vector from input
    
    // Allocate CPU variables in the host memory
    Real *a = new Real[n];
    Real *b = new Real[n];
    Real *c_GPU = new Real[n];
    Real *c_CPU = new Real[n]; 


    // Initialize arrays on CPU
    for (int i = 0; i < n; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // Calculating in cpu first
    time_t getTime;
    getTime = clock();
    addCPU ( a, b, c_CPU, n);
    getTime = clock() - getTime;
    double CPUTime = 1000.0*getTime/ CLOCKS_PER_SEC;
    printf("CPU addition time: %.3f ms\n\n",CPUTime);

    // Allocate GPU variables in the device memory
    Real *a_d, *b_d, *c_d;

    HANDLE_ERROR(cudaMalloc((void**)&a_d, n * sizeof(Real)));
    HANDLE_ERROR(cudaMalloc((void**)&b_d, n * sizeof(Real)));
    HANDLE_ERROR(cudaMalloc((void**)&c_d, n * sizeof(Real)));

    // Copy data from host to device
    HANDLE_ERROR(cudaMemcpy(a_d, a, n * sizeof(Real), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(b_d, b, n * sizeof(Real), cudaMemcpyHostToDevice));

    // Print header
    printf("%-12s %-6s %-12s %-12s %-12s %-10s %-10s\n", 
           "Nb", "Nt", "Nb*Nt", "n", "maxErr", "GPU(ms)", "speedup");
    
    // Array of thread counts 
    int threads[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int numTests = sizeof(threads)/sizeof(threads[0]);

    for ( int t = 0; t < numTests; t++)
    {
        int Nt = threads[t];
        int Nb = ceil((1.*n)/Nt);

        // Using CUDA event for GPU timing
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));

        // Record start event
        HANDLE_ERROR(cudaEventRecord(start));

        // Execute in GPU
        add <<<Nb, Nt>>> (a_d, b_d, c_d, n);

        // Record stop event
        HANDLE_ERROR(cudaEventRecord(stop));

        // Wait for the stop event to complete
        HANDLE_ERROR(cudaEventSynchronize(stop));
        
        // Calculate elapsed time
        float GPUTime = 0;
        HANDLE_ERROR(cudaEventElapsedTime(&GPUTime, start, stop));

        // Check for the launch errors
        HANDLE_ERROR(cudaGetLastError());

        // Copy result from device to host
        HANDLE_ERROR(cudaMemcpy(c_GPU, c_d, n * sizeof(Real), cudaMemcpyDeviceToHost));
        
        // Compute max norm error
        Real maxErr = computeMaxNorm(c_GPU, c_CPU, n);
        
        // Calculate speedup
        double speedup = CPUTime / GPUTime;

        // Print results
        printf("%-12d %-6d %-12d %-12d %.2e   %-10.3f %-10.2f\n", 
               Nb, Nt, Nb*Nt, n, maxErr, GPUTime, speedup);
        
        // Destroy events
        HANDLE_ERROR(cudaEventDestroy(start));
        HANDLE_ERROR(cudaEventDestroy(stop));
    }

    // Clean the memory after use
    HANDLE_ERROR(cudaFree(a_d));
    HANDLE_ERROR(cudaFree(b_d));
    HANDLE_ERROR(cudaFree(c_d));

    delete[] a;
    delete[] b;
    delete[] c_CPU;
    delete[] c_GPU;
    
    return 0;
}