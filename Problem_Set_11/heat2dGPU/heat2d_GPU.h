#ifndef HEAT2D_GPU_H
#define HEAT2D_GPU_H

// Boundary condition types
#define BC_DIRICHLET 1
#define BC_NEUMANN 2

// Solution types
#define TRIG_DD 1
#define MANUFACTURED 2

// Integer parameter indices
#define IPAR_NX 0
#define IPAR_NY 1
#define IPAR_N1A 2
#define IPAR_N1B 3
#define IPAR_N2A 4
#define IPAR_N2B 5
#define IPAR_ND1A 6
#define IPAR_ND1B 7
#define IPAR_ND2A 8
#define IPAR_ND2B 9
#define IPAR_ND1 10
#define IPAR_ND2 11
#define IPAR_BC_LEFT 12
#define IPAR_BC_RIGHT 13
#define IPAR_BC_BOTTOM 14
#define IPAR_BC_TOP 15
#define IPAR_SOLUTION 16
#define IPAR_SIZE 17

// Real parameter indices
#define RPAR_XA 0
#define RPAR_XB 1
#define RPAR_YA 2
#define RPAR_YB 3
#define RPAR_DX 4
#define RPAR_DY 5
#define RPAR_DT 6
#define RPAR_KAPPA 7
#define RPAR_RX 8
#define RPAR_RY 9
#define RPAR_KX 10
#define RPAR_KY 11
#define RPAR_KXPI 12
#define RPAR_KYPI 13
#define RPAR_KAPPA_K2 14
#define RPAR_B0 15
#define RPAR_B1 16
#define RPAR_B2 17
#define RPAR_C0 18
#define RPAR_C1 19
#define RPAR_C2 20
#define RPAR_A0 21
#define RPAR_A1 22
#define RPAR_A2 23
#define RPAR_SIZE 24

// Function declaration
typedef double Real;

Real SolveHeat2dGPU(Real *u_initial, int numSteps, Real tFinal, 
                    int *ipar_h, Real *rpar_h, Real *x_h, Real *y_h,
                    Real &cpuTime, int &numThreads, int debug);

#endif // HEAT2D_GPU_H