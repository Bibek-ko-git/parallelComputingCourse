/**
 * @file navierstokes.h
 * @brief 2D Incompressible Navier-Stokes Solver using Projection Method
 */
#ifndef NAVIERSTOKES_H
#define NAVIERSTOKES_H

#include <functional>

typedef double Real;

class NavierStokes {
public:
    // Grid parameters
    int Nx, Ny;
    Real Lx, Ly;
    Real dx, dy;
    
    // Physical parameters
    Real rho, nu;
    
    // Field variables
    Real* u;        // x-velocity
    Real* v;        // y-velocity
    Real* p;        // pressure
    Real* u_old;    // previous x-velocity
    Real* v_old;    // previous y-velocity
    Real* rhs_u;    // momentum RHS (x)
    Real* rhs_v;    // momentum RHS (y)
    Real* rhs_p;    // pressure Poisson RHS
    
    // Coordinates
    Real* x;
    Real* y;
    
    // Time
    Real current_time;
    
    // Timing (for performance analysis)
    double time_pressure;
    double time_velocity;
    double time_rhs;
    int pressure_iterations;
    
    // Callbacks
    std::function<void(Real*, Real*, Real*, Real, int, int, Real, Real)> bc_function;
    std::function<void(Real, Real, Real, Real, Real, Real&, Real&)> source_function;
    
    // Constructor/Destructor
    NavierStokes(int Nx, int Ny, Real Lx, Real Ly, Real rho, Real nu);
    ~NavierStokes();
    
    // Setup
    void setInitialConditions(
        std::function<Real(Real, Real)> u_init,
        std::function<Real(Real, Real)> v_init,
        std::function<Real(Real, Real)> p_init);
    void setBoundaryConditions(
        std::function<void(Real*, Real*, Real*, Real, int, int, Real, Real)> bc_func);
    void setSourceTerm(
        std::function<void(Real, Real, Real, Real, Real, Real&, Real&)> source_func);
    
    // Indexing
    inline int idx(int i, int j) const { return i + j * (Nx + 1); }
    
    // Spatial operators
    Real du_dx(int i, int j);
    Real du_dy(int i, int j);
    Real dv_dx(int i, int j);
    Real dv_dy(int i, int j);
    Real dp_dx(int i, int j);
    Real dp_dy(int i, int j);
    Real divergence(int i, int j);
    Real laplacian_u(int i, int j);
    Real laplacian_v(int i, int j);
    Real convection_u(int i, int j);
    Real convection_v(int i, int j);
    
    // Solvers
    void solvePressurePoissonGS(int max_iter = 5000, Real tol = 1e-6);
    void applyPressureBC();
    void timeStepUpdate(Real dt);
    Real computeDivergenceNorm();
    
    // Driver
    void solve(Real t_final, Real dt, int output_interval);
    void writeVTK(const char* filename, Real time);
    
    // Timing
    void resetTimers();
    void printTimingStats();
};

#endif