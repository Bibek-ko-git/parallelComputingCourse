//
// Header file for CUDA-based 1D heat equation solver
//

#ifndef HEAT1D_GPU_H
#define HEAT1D_GPU_H

typedef double Real;

// Integer parameter indices for ipar array
#define IPAR_NX       0
#define IPAR_N1A      1
#define IPAR_N1B      2
#define IPAR_ND1A     3
#define IPAR_ND1B     4
#define IPAR_ND1      5
#define IPAR_BC_LEFT  6
#define IPAR_BC_RIGHT 7
#define IPAR_SOLUTION 8
#define IPAR_SIZE     9

// Real parameter indices for rpar array
#define RPAR_XA       0
#define RPAR_XB       1
#define RPAR_DX       2
#define RPAR_DX2      3
#define RPAR_DT       4
#define RPAR_KAPPA    5
#define RPAR_RX       6
#define RPAR_KX       7
#define RPAR_KXPI     8
#define RPAR_KAPPOPISQ 9
#define RPAR_B0      10
#define RPAR_B1      11
#define RPAR_B2      12
#define RPAR_A0      13
#define RPAR_A1      14
#define RPAR_SIZE    15

// Solution type constants
#define TRIG_DD 1
#define TRIG_NN 2
#define POLY_DD 3
#define POLY_NN 4

// Boundary condition constants
#define BC_DIRICHLET 1
#define BC_NEUMANN   2

//
// Main GPU solver function
// Returns: maximum error at final time
//
Real SolveHeat1dGPU(
    Real *u_initial,      // Initial condition (host array)
    int numSteps,         // Number of time steps
    Real tFinal,          // Final time
    int *ipar_h,          // Integer parameters (host array)
    Real *rpar_h,         // Real parameters (host array)
    Real *x_h,            // Grid points (host array)
    Real &cpuTime,        // Output: CPU time for GPU computation
    int &numThreads,      // Copy number of threads from GPU to CPU
    int debug             // Debug flag on or off
);

#endif // HEAT1D_GPU_H