/**
 * @file TestCases.C
 * @brief Test cases for Navier-Stokes solver (Projection Method)
 */
#include "TestCases.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace TestCases {

// ==================================================================================
// Lid-Driven Cavity
// ==================================================================================
namespace Cavity {

void boundary_conditions(Real* u, Real* v, Real* p, Real t,
                         int Nx, int Ny, Real dx, Real dy) {
    (void)p; (void)t; (void)dx; (void)dy;
    
    auto idx = [Nx](int i, int j) { return i + j * (Nx + 1); };
    
    // Bottom wall: u=v=0
    for (int i = 0; i <= Nx; ++i) {
        u[idx(i, 0)] = 0.0;
        v[idx(i, 0)] = 0.0;
    }
    
    // Top wall: u=1, v=0 (moving lid)
    for (int i = 0; i <= Nx; ++i) {
        u[idx(i, Ny)] = 1.0;
        v[idx(i, Ny)] = 0.0;
    }
    
    // Left wall: u=v=0
    for (int j = 0; j <= Ny; ++j) {
        u[idx(0, j)] = 0.0;
        v[idx(0, j)] = 0.0;
    }
    
    // Right wall: u=v=0
    for (int j = 0; j <= Ny; ++j) {
        u[idx(Nx, j)] = 0.0;
        v[idx(Nx, j)] = 0.0;
    }
}

void run(int N, Real t_final) {
    std::cout << "\n=== Lid-Driven Cavity ===\n";
    std::cout << "Grid: " << (N+1) << " x " << (N+1) << "\n";
    
    Real L = 1.0;
    Real Re = 100.0;
    Real U_lid = 1.0;
    Real nu = U_lid * L / Re;
    Real rho = 1.0;
    Real h = L / N;
    
    Real dt_cfl = 0.9 * h / U_lid;
    Real dt_diff = 0.9 * h * h / (4*nu); 
    Real dt = std::min(dt_cfl, dt_diff);
    int num_steps = static_cast<int>(std::ceil(t_final / dt));
    
    std::cout << "Re = " << Re << "\n";
    std::cout << "dt = " << dt << ", t_final = " << t_final << "\n";
    std::cout << "Steps = " << num_steps << "\n";
    
    NavierStokes solver(N, N, L, L, rho, nu);
    
    solver.setInitialConditions(
        [](Real, Real) { return 0.0; },
        [](Real, Real) { return 0.0; },
        [](Real, Real) { return 0.0; }
    );
    
    solver.setBoundaryConditions(boundary_conditions);
    
    // Apply BC to initial condition
    boundary_conditions(solver.u, solver.v, solver.p, 0.0, N, N, h, h);
    
    int output_interval = std::max(1, num_steps / 50);
    solver.solve(t_final, dt, output_interval);
}

} // namespace Cavity

// ==================================================================================
// Method of Manufactured Solutions (MMS)
// ==================================================================================
namespace MMS {

const Real PI = 3.14159265358979323846;

// Exact solution (steady)
Real u_exact(Real x, Real y, Real) {
    return std::sin(PI * x) * std::cos(PI * y);
}

Real v_exact(Real x, Real y, Real) {
    return -std::cos(PI * x) * std::sin(PI * y);
}

Real p_exact(Real x, Real y, Real) {
    return std::sin(PI * x) * std::sin(PI * y);
}

// Source term to force exact solution
// NOTE: For projection method, the pressure gradient is applied separately,
// so the source term should only balance convection and diffusion
void source_term(Real x, Real y, Real t, Real nu, Real rho, Real& fx, Real& fy) {
    (void)t; (void)rho;  // pressure handled by projection
    
    Real u = u_exact(x, y, t);
    Real v = v_exact(x, y, t);
    
    Real ux = PI * std::cos(PI * x) * std::cos(PI * y);
    Real uy = -PI * std::sin(PI * x) * std::sin(PI * y);
    Real vx = PI * std::sin(PI * x) * std::sin(PI * y);
    Real vy = -PI * std::cos(PI * x) * std::cos(PI * y);
    
    Real lap_u = -2.0 * PI * PI * u;
    Real lap_v = -2.0 * PI * PI * v;
    
    // Balance convection and diffusion only (projection handles pressure)
    fx = u * ux + v * uy - nu * lap_u;
    fy = u * vx + v * vy - nu * lap_v;
}

void boundary_conditions(Real* u, Real* v, Real* p, Real t,
                         int Nx, int Ny, Real dx, Real dy) {
    (void)p;
    auto idx = [Nx](int i, int j) { return i + j * (Nx + 1); };
    
    for (int i = 0; i <= Nx; ++i) {
        Real x = i * dx;
        u[idx(i, 0)] = u_exact(x, 0.0, t);
        v[idx(i, 0)] = v_exact(x, 0.0, t);
        u[idx(i, Ny)] = u_exact(x, 1.0, t);
        v[idx(i, Ny)] = v_exact(x, 1.0, t);
    }
    for (int j = 0; j <= Ny; ++j) {
        Real y = j * dy;
        u[idx(0, j)] = u_exact(0.0, y, t);
        v[idx(0, j)] = v_exact(0.0, y, t);
        u[idx(Nx, j)] = u_exact(1.0, y, t);
        v[idx(Nx, j)] = v_exact(1.0, y, t);
    }
}

void run(int N, Real t_final) {
    std::cout << "\n=== Method of Manufactured Solutions ===\n";
    std::cout << "Grid: " << (N+1) << " x " << (N+1) << "\n";
    
    Real L = 1.0;
    Real nu = 0.01;
    Real rho = 1.0;
    Real h = L / N;
    

    Real dt = 0.9 * h * h / (4*nu); 
    int num_steps = static_cast<int>(std::ceil(t_final / dt));
    
    std::cout << "nu = " << nu << "\n";
    std::cout << "dt = " << dt << ", t_final = " << t_final << "\n";
    std::cout << "Steps = " << num_steps << "\n";
    
    NavierStokes solver(N, N, L, L, rho, nu);
    
    solver.setInitialConditions(
        [](Real x, Real y) { return u_exact(x, y, 0); },
        [](Real x, Real y) { return v_exact(x, y, 0); },
        [](Real x, Real y) { return p_exact(x, y, 0); }
    );
    
    solver.setBoundaryConditions(boundary_conditions);
    solver.setSourceTerm(source_term);
    
    int output_interval = std::max(1, num_steps / 50);
    solver.solve(t_final, dt, output_interval);
    
    // Compute velocity errors (pressure not meaningful in projection method)
    Real u_err = 0.0, v_err = 0.0;
    Real u_norm = 0.0, v_norm = 0.0;
    int count = 0;
    
    for (int j = 1; j < N; ++j) {
        for (int i = 1; i < N; ++i) {
            Real x = i * h, y = j * h;
            int idx = i + j * (N + 1);
            
            Real u_ex = u_exact(x, y, t_final);
            Real v_ex = v_exact(x, y, t_final);
            
            u_err += std::pow(solver.u[idx] - u_ex, 2);
            v_err += std::pow(solver.v[idx] - v_ex, 2);
            
            u_norm += u_ex * u_ex;
            v_norm += v_ex * v_ex;
            
            count++;
        }
    }
    
    std::cout << "\n=== Final L2 Errors ===\n";
    std::cout << "Relative L2(u) = " << std::scientific << std::setprecision(4) 
              << std::sqrt(u_err / u_norm) << "\n";
    std::cout << "Relative L2(v) = " << std::sqrt(v_err / v_norm) << "\n";
}

} // namespace MMS

// ==================================================================================
// Poiseuille Flow (channel flow between parallel plates driven by pressure gradient)
// ==================================================================================
namespace Poiseuille {

// Flow parameters (defined at namespace level for access by all functions)
const Real Lx = 4.0;       // Channel length
const Real Ly = 1.0;       // Channel height (H)
const Real U_max = 1.0;    // Maximum velocity at centerline
const Real nu = 0.01;      // Kinematic viscosity
const Real rho = 1.0;      // Density

// Derived quantities
const Real dpdx = -8.0 * nu * U_max / (Ly * Ly);  // Pressure gradient (negative = flow in +x)
const Real p_inlet = -dpdx * Lx;                   // Pressure at inlet (outlet is 0)

// Exact solutions
Real u_exact(Real x, Real y) {
    (void)x;  // u doesn't depend on x for fully developed flow
    return 4.0 * U_max * y * (Ly - y) / (Ly * Ly);
}

Real v_exact(Real x, Real y) {
    (void)x; (void)y;
    return 0.0;
}

Real p_exact(Real x, Real y) {
    (void)y;  // p doesn't depend on y (no vertical pressure gradient)
    return p_inlet + dpdx * x;  // Linear pressure drop from inlet to outlet
}

// Source term: body force equivalent to pressure gradient
// This drives the flow since we use projection method
void source_term(Real x, Real y, Real t, Real nu_loc, Real rho_loc, Real& fx, Real& fy) {
    (void)x; (void)y; (void)t; (void)nu_loc; (void)rho_loc;
    fx = -dpdx / rho; 
    fy = 0.0;
}

void boundary_conditions(Real* u, Real* v, Real* p, Real t,
                         int Nx, int Ny, Real dx, Real dy) {
    (void)p; (void)t; (void)dx; 
    auto idx = [Nx](int i, int j) { return i + j * (Nx + 1); };
    
    Real Ly_local = Ny * dy;
    
    // Top and bottom walls: no-slip
    for (int i = 0; i <= Nx; ++i) {
        u[idx(i, 0)] = 0.0;
        v[idx(i, 0)] = 0.0;
        u[idx(i, Ny)] = 0.0;
        v[idx(i, Ny)] = 0.0;
    }
    
    // Inlet (left): parabolic profile (Dirichlet)
    for (int j = 1; j < Ny; ++j) {
        Real y = j * dy;
        u[idx(0, j)] = 4.0 * U_max * y * (Ly_local - y) / (Ly_local * Ly_local);
        v[idx(0, j)] = 0.0;
    }
    
    // Outlet (right): parabolic profile (Dirichlet) - fully developed
    for (int j = 0; j <= Ny; ++j) {
        Real y = j * dy;
        u[idx(Nx, j)] = 4.0 * U_max * y * (Ly_local - y) / (Ly_local * Ly_local);
        v[idx(Nx, j)] = 0.0;
    }
    
    
}

void run(int N, Real t_final) {
    std::cout << "\n=== Poiseuille Flow (Channel) ===\n";
    
    int Nx = 4 * N, Ny = N;
    Real hx = Lx / Nx;
    Real hy = Ly / Ny;
    Real h = std::min(hx, hy);
    
    std::cout << "Grid: " << (Nx+1) << " x " << (Ny+1) << "\n";
    std::cout << "Channel: Lx = " << Lx << ", Ly = " << Ly << "\n";
    std::cout << "nu = " << nu << ", Re = " << U_max * Ly / nu << "\n";
    std::cout << "Pressure gradient: dp/dx = " << dpdx << "\n";
    
    // Time step based on CFL
    Real dt_cfl = 0.9 * h / U_max;
    Real dt_diff = 0.9 * h * h / (4*nu); 
    Real dt = std::min(dt_cfl, dt_diff);
    int num_steps = static_cast<int>(std::ceil(t_final / dt));
    
    std::cout << "dt = " << dt << ", t_final = " << t_final << "\n";
    std::cout << "Steps = " << num_steps << "\n";
    
    NavierStokes solver(Nx, Ny, Lx, Ly, rho, nu);
    
    // Initialize with exact solution (avoids startup transient)
    solver.setInitialConditions(
        [](Real x, Real y) { return 0.0; },
        [](Real x, Real y) { return 0.0; },
        [](Real x, Real y) { return 0.0; }
    );
    
    solver.setBoundaryConditions(boundary_conditions);
    solver.setSourceTerm(source_term);
    
    int output_interval = std::max(1, num_steps / 50);
    solver.solve(t_final, dt, output_interval);
    
    // ============================================================
    // Compute errors against exact solution
    // ============================================================
    Real u_err = 0.0, v_err = 0.0;
    Real u_norm = 0.0, v_norm = 0.0;
    Real u_max_err = 0.0, v_max_err = 0.0;
    int count = 0;
    
    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            Real x = i * hx;
            Real y = j * hy;
            int idx = i + j * (Nx + 1);
            
            Real u_ex = u_exact(x, y);
            Real v_ex = v_exact(x, y);
            
            Real err_u = solver.u[idx] - u_ex;
            Real err_v = solver.v[idx] - v_ex;
            
            u_err += err_u * err_u;
            v_err += err_v * err_v;
            
            u_norm += u_ex * u_ex;
            v_norm += v_ex * v_ex;
            
            u_max_err = std::max(u_max_err, std::abs(err_u));
            v_max_err = std::max(v_max_err, std::abs(err_v));
            
            count++;
        }
    }
    
    std::cout << "\n=== Errors vs Exact Poiseuille Solution ===\n";
    std::cout << "Relative L2(u) = " << std::scientific << std::setprecision(4) 
              << std::sqrt(u_err / u_norm) << "\n";
    if (v_norm > 1e-14) {
        std::cout << "Relative L2(v) = " << std::sqrt(v_err / v_norm) << "\n";
    } else {
        std::cout << "Absolute L2(v) = " << std::sqrt(v_err / count) << "\n";
    }
    std::cout << "Max |u - u_exact| = " << u_max_err << "\n";
    std::cout << "Max |v - v_exact| = " << v_max_err << "\n";
    
    // Print centerline velocity profile
    std::cout << "\n=== Centerline u-velocity (x = Lx/2) ===\n";
    int i_mid = Nx / 2;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "    y        u_num      u_exact     error\n";
    for (int j = 0; j <= Ny; j += std::max(1, Ny/8)) {
        Real y = j * hy;
        int idx = i_mid + j * (Nx + 1);
        Real u_ex = u_exact(i_mid * hx, y);
        std::cout << std::setw(8) << y 
                  << std::setw(12) << solver.u[idx]
                  << std::setw(12) << u_ex
                  << std::setw(12) << std::scientific << (solver.u[idx] - u_ex) << "\n";
        std::cout << std::fixed;
    }
}

} // namespace Poiseuille

} // namespace TestCases
