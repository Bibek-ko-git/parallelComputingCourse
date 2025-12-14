#ifndef NAVIERSTOKES_H
#define NAVIERSTOKES_H

#include <functional>
#include <mpi.h>

using Real = double;

class NavierStokes {
public:
    int Nx, Ny;
    Real Lx, Ly;
    Real rho, nu;
    
    int ny_local;     
    int j_start, j_end;     
    
    int rank, size;
    int n_neighbor, s_neighbor;

    Real dx, dy;
    Real current_time;

    Real *u, *v, *p;
    Real *u_old, *v_old;
    Real *rhs_u, *rhs_v, *rhs_p;
    
    Real *x;       
    Real *y_local; 

    double time_pressure, time_velocity, time_rhs;
    long long pressure_iterations;

    std::function<void(Real*, Real*, Real*, Real, int, int, Real, Real)> bc_function;
    std::function<void(Real, Real, Real, Real, Real, Real&, Real&)> source_function;

public:
    NavierStokes(int Nx, int Ny, Real Lx, Real Ly, Real rho, Real nu);
    ~NavierStokes();

    void setInitialConditions(std::function<Real(Real, Real)> u_init,
                              std::function<Real(Real, Real)> v_init,
                              std::function<Real(Real, Real)> p_init);
    
    void setBoundaryConditions(std::function<void(Real*, Real*, Real*, Real, int, int, Real, Real)> bc_func);
    void setSourceTerm(std::function<void(Real, Real, Real, Real, Real, Real&, Real&)> source_func);

    void solve(Real t_final, Real dt, int output_interval);
    void writeVTK(const char* filename, Real time);

    void resetTimers();
    void printTimingStats();
    int idx(int i, int j) const { return i + j * (Nx + 1); }

private:
    void timeStepUpdate(Real dt);
    void applyPressureBC();
    void solvePressurePoissonGS(int max_iter, Real tol_abs);
    Real computeDivergenceNorm();
    void exchangeHalos(Real* data);

    inline Real dp_dx(int i, int j);
    inline Real dp_dy(int i, int j);
    inline Real du_dx(int i, int j);
    inline Real du_dy(int i, int j);
    inline Real dv_dx(int i, int j);
    inline Real dv_dy(int i, int j);
    inline Real divergence(int i, int j);
    inline Real laplacian_u(int i, int j);
    inline Real laplacian_v(int i, int j);
    inline Real convection_u(int i, int j);
    inline Real convection_v(int i, int j);
};
#endif
