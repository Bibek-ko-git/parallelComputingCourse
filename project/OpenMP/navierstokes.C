/**
 * @file navierstokes.C
 * @brief 2D Incompressible Navier-Stokes Solver using Projection Method
 *        OpenMP parallelized version
 */
#include "navierstokes.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <omp.h>

// ==================================================================================
// Constructor
// ==================================================================================
NavierStokes::NavierStokes(int Nx, int Ny, Real Lx, Real Ly, Real rho, Real nu)
    : Nx(Nx), Ny(Ny), Lx(Lx), Ly(Ly), rho(rho), nu(nu), current_time(0.0),
      time_pressure(0.0), time_velocity(0.0), time_rhs(0.0), pressure_iterations(0)
{
    dx = Lx / Nx;
    dy = Ly / Ny;

    int total_nodes = (Nx + 1) * (Ny + 1);

    // Allocate and zero-initialize arrays
    u = new Real[total_nodes]();
    v = new Real[total_nodes]();
    p = new Real[total_nodes]();
    u_old = new Real[total_nodes]();
    v_old = new Real[total_nodes]();
    rhs_u = new Real[total_nodes]();
    rhs_v = new Real[total_nodes]();
    rhs_p = new Real[total_nodes]();

    x = new Real[Nx + 1];
    y = new Real[Ny + 1];
    for (int i = 0; i <= Nx; ++i) x[i] = i * dx;
    for (int j = 0; j <= Ny; ++j) y[j] = j * dy;

    std::cout << "NavierStokes2D (OpenMP) initialized:\n";
    std::cout << "  Grid: " << (Nx+1) << " x " << (Ny+1) << "\n";
    std::cout << "  Cell size: dx = " << dx << ", dy = " << dy << "\n";
    std::cout << "  Domain: [0, " << Lx << "] x [0, " << Ly << "]\n";
    std::cout << "  rho = " << rho << ", nu = " << nu << "\n";
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << "\n";
}

// ==================================================================================
// Destructor
// ==================================================================================
NavierStokes::~NavierStokes() {
    delete[] u;
    delete[] v;
    delete[] p;
    delete[] u_old;
    delete[] v_old;
    delete[] rhs_u;
    delete[] rhs_v;
    delete[] rhs_p;
    delete[] x;
    delete[] y;
}

// ==================================================================================
// Timing functions
// ==================================================================================
void NavierStokes::resetTimers() {
    time_pressure = 0.0;
    time_velocity = 0.0;
    time_rhs = 0.0;
    pressure_iterations = 0;
}

void NavierStokes::printTimingStats() {
    double total = time_pressure + time_velocity + time_rhs;
    std::cout << "\n=== Timing Statistics ===\n";
    std::cout << "  Pressure solve: " << std::fixed << std::setprecision(3) 
              << time_pressure << "s (" << 100.0*time_pressure/total << "%)\n";
    std::cout << "  Velocity update: " << time_velocity << "s (" 
              << 100.0*time_velocity/total << "%)\n";
    std::cout << "  RHS computation: " << time_rhs << "s (" 
              << 100.0*time_rhs/total << "%)\n";
    std::cout << "  Total: " << total << "s\n";
    std::cout << "  Avg pressure iterations: " << pressure_iterations << "\n";
}

// ==================================================================================
// Setup Functions
// ==================================================================================
void NavierStokes::setInitialConditions(
    std::function<Real(Real, Real)> u_init,
    std::function<Real(Real, Real)> v_init,
    std::function<Real(Real, Real)> p_init)
{
    #pragma omp parallel for schedule(static)
    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            int index = idx(i, j);
            u[index] = u_init(x[i], y[j]);
            v[index] = v_init(x[i], y[j]);
            p[index] = p_init(x[i], y[j]);
        }
    }
}

void NavierStokes::setBoundaryConditions(
    std::function<void(Real*, Real*, Real*, Real, int, int, Real, Real)> bc_func)
{
    bc_function = bc_func;
}

void NavierStokes::setSourceTerm(
    std::function<void(Real, Real, Real, Real, Real, Real&, Real&)> source_func)
{
    source_function = source_func;
}

// ==================================================================================
// Spatial Operators (2nd-order central differences)
// ==================================================================================
Real NavierStokes::dp_dx(int i, int j) {
    return (p[idx(i+1,j)] - p[idx(i-1,j)]) / (2.0 * dx);
}

Real NavierStokes::dp_dy(int i, int j) {
    return (p[idx(i,j+1)] - p[idx(i,j-1)]) / (2.0 * dy);
}

Real NavierStokes::du_dx(int i, int j) {
    return (u[idx(i+1,j)] - u[idx(i-1,j)]) / (2.0 * dx);
}

Real NavierStokes::du_dy(int i, int j) {
    return (u[idx(i,j+1)] - u[idx(i,j-1)]) / (2.0 * dy);
}

Real NavierStokes::dv_dx(int i, int j) {
    return (v[idx(i+1,j)] - v[idx(i-1,j)]) / (2.0 * dx);
}

Real NavierStokes::dv_dy(int i, int j) {
    return (v[idx(i,j+1)] - v[idx(i,j-1)]) / (2.0 * dy);
}

Real NavierStokes::divergence(int i, int j) {
    return du_dx(i, j) + dv_dy(i, j);
}

Real NavierStokes::laplacian_u(int i, int j) {
    Real d2u_dx2 = (u[idx(i+1,j)] - 2.0*u[idx(i,j)] + u[idx(i-1,j)]) / (dx * dx);
    Real d2u_dy2 = (u[idx(i,j+1)] - 2.0*u[idx(i,j)] + u[idx(i,j-1)]) / (dy * dy);
    return d2u_dx2 + d2u_dy2;
}

Real NavierStokes::laplacian_v(int i, int j) {
    Real d2v_dx2 = (v[idx(i+1,j)] - 2.0*v[idx(i,j)] + v[idx(i-1,j)]) / (dx * dx);
    Real d2v_dy2 = (v[idx(i,j+1)] - 2.0*v[idx(i,j)] + v[idx(i,j-1)]) / (dy * dy);
    return d2v_dx2 + d2v_dy2;
}

Real NavierStokes::convection_u(int i, int j) {
    return u[idx(i,j)] * du_dx(i,j) + v[idx(i,j)] * du_dy(i,j);
}

Real NavierStokes::convection_v(int i, int j) {
    return u[idx(i,j)] * dv_dx(i,j) + v[idx(i,j)] * dv_dy(i,j);
}

// ==================================================================================
// Pressure Boundary Conditions
// ==================================================================================
void NavierStokes::applyPressureBC() {
    // Left/Right boundaries
    #pragma omp parallel for
    for (int j = 1; j < Ny; ++j) {
        p[idx(0, j)]  = p[idx(1, j)];
        p[idx(Nx, j)] = p[idx(Nx-1, j)];
    }
    
    // Bottom/Top boundaries
    #pragma omp parallel for
    for (int i = 1; i < Nx; ++i) {
        p[idx(i, 0)]  = p[idx(i, 1)];
        p[idx(i, Ny)] = p[idx(i, Ny-1)];
    }
    
    // Corners (average of neighbors)
    p[idx(0, 0)]   = 0.5 * (p[idx(1, 0)]   + p[idx(0, 1)]);
    p[idx(Nx, 0)]  = 0.5 * (p[idx(Nx-1, 0)] + p[idx(Nx, 1)]);
    p[idx(0, Ny)]  = 0.5 * (p[idx(1, Ny)]  + p[idx(0, Ny-1)]);
    p[idx(Nx, Ny)] = 0.5 * (p[idx(Nx-1, Ny)] + p[idx(Nx, Ny-1)]);
}

// ==================================================================================
// Pressure Poisson Solver (Red-Black Gauss-Seidel with SOR using OpenMP)
// ==================================================================================
void NavierStokes::solvePressurePoissonGS(int max_iter, Real tol) {
    Real ax = 1.0 / (dx * dx);
    Real ay = 1.0 / (dy * dy);
    Real ap = -2.0 * (ax + ay);
    Real inv_ap = 1.0 / ap;
   
    // SOR relaxation factor
    const Real omega = 1.99;
    
    int iter;
    for (iter = 0; iter < max_iter; ++iter) {
        // Red sweep (i+j even) - parallelized over rows, with SOR
        #pragma omp parallel for schedule(static)
        for (int j = 1; j < Ny; ++j) {
            int i_start = 1 + (j % 2);  // Start at red point
            for (int i = i_start; i < Nx; i += 2) {
                int index = idx(i, j);
                Real p_gs = (rhs_p[index] 
                           - ax * (p[idx(i+1,j)] + p[idx(i-1,j)])
                           - ay * (p[idx(i,j+1)] + p[idx(i,j-1)])) * inv_ap;
                p[index] = (1.0 - omega) * p[index] + omega * p_gs;
            }
        }
                
        // Black sweep (i+j odd) - parallelized over rows, with SOR
        #pragma omp parallel for schedule(static)
        for (int j = 1; j < Ny; ++j) {
            int i_start = 1 + ((j + 1) % 2);  // Start at black point
            for (int i = i_start; i < Nx; i += 2) {
                int index = idx(i, j);
                Real p_gs = (rhs_p[index] 
                           - ax * (p[idx(i+1,j)] + p[idx(i-1,j)])
                           - ay * (p[idx(i,j+1)] + p[idx(i,j-1)])) * inv_ap;
                p[index] = (1.0 - omega) * p[index] + omega * p_gs;
            }
        }
        
        // Apply Neumann BCs
        applyPressureBC();
        
        // Check convergence every 10 iterations
        if (iter % 10 == 0 || iter == max_iter - 1) {
            Real max_res = 0.0;
            #pragma omp parallel for reduction(max:max_res)
            for (int j = 1; j < Ny; ++j) {
                for (int i = 1; i < Nx; ++i) {
                    Real lap = ax * (p[idx(i+1,j)] - 2.0*p[idx(i,j)] + p[idx(i-1,j)]) +
                               ay * (p[idx(i,j+1)] - 2.0*p[idx(i,j)] + p[idx(i,j-1)]);
                    Real res = std::abs(lap - rhs_p[idx(i,j)]);
                    max_res = std::max(max_res, res);
                }
            }
            
            if (max_res < tol) {
                pressure_iterations += iter + 1;
                return;
            }
        }
    }
    pressure_iterations += max_iter;
}

// ==================================================================================
// Time Step Update (Projection Method)
// ==================================================================================
void NavierStokes::timeStepUpdate(Real dt) {
    double t_start;
    
    // Save old velocities
    std::memcpy(u_old, u, sizeof(Real) * (Nx + 1) * (Ny + 1));
    std::memcpy(v_old, v, sizeof(Real) * (Nx + 1) * (Ny + 1));
    
    // ============================================================
    // STEP 1: Intermediate velocity (explicit convection + diffusion)
    // ============================================================
    t_start = omp_get_wtime();
    
    #pragma omp parallel for schedule(static)
    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            int index = idx(i, j);
            
            Real fx = 0.0, fy = 0.0;
            if (source_function) {
                source_function(x[i], y[j], current_time, nu, rho, fx, fy);
            }
            
            rhs_u[index] = -convection_u(i,j) + nu * laplacian_u(i,j) + fx;
            rhs_v[index] = -convection_v(i,j) + nu * laplacian_v(i,j) + fy;
        }
    }
    
    #pragma omp parallel for schedule(static)
    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            int index = idx(i, j);
            u[index] = u_old[index] + dt * rhs_u[index];
            v[index] = v_old[index] + dt * rhs_v[index];
        }
    }
    
    time_rhs += omp_get_wtime() - t_start;
    
    // Apply velocity BCs to intermediate velocity
    if (bc_function) {
        bc_function(u, v, p, current_time + dt, Nx, Ny, dx, dy);
    }
    
    // ============================================================
    // STEP 2: Pressure Poisson equation
    // ============================================================
    t_start = omp_get_wtime();
    
    #pragma omp parallel for schedule(static)
    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            rhs_p[idx(i,j)] = (rho / dt) * divergence(i,j);
        }
    }
    
    // Remove mean from RHS (solvability condition for Neumann BC)
    Real rhs_sum = 0.0;
    #pragma omp parallel for reduction(+:rhs_sum)
    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            rhs_sum += rhs_p[idx(i,j)];
        }
    }
    Real rhs_mean = rhs_sum / ((Nx - 1) * (Ny - 1));
    
    #pragma omp parallel for schedule(static)
    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            rhs_p[idx(i,j)] -= rhs_mean;
        }
    }
    
    time_rhs += omp_get_wtime() - t_start;
    
    // Solve pressure Poisson
    t_start = omp_get_wtime();
    solvePressurePoissonGS(5000, 1e-6);
    time_pressure += omp_get_wtime() - t_start;
    
    // Remove mean from pressure solution
    Real p_sum = 0.0;
    #pragma omp parallel for reduction(+:p_sum)
    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            p_sum += p[idx(i,j)];
        }
    }
    Real p_mean = p_sum / ((Nx + 1) * (Ny + 1));
    
    #pragma omp parallel for schedule(static)
    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            p[idx(i,j)] -= p_mean;
        }
    }
    
    applyPressureBC();
    
    // ============================================================
    // STEP 3: Velocity correction (projection)
    // ============================================================
    t_start = omp_get_wtime();
    
    #pragma omp parallel for schedule(static)
    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            int index = idx(i, j);
            u[index] -= (dt / rho) * dp_dx(i, j);
            v[index] -= (dt / rho) * dp_dy(i, j);
        }
    }
    
    time_velocity += omp_get_wtime() - t_start;
    
    // Apply velocity BCs after correction
    if (bc_function) {
        bc_function(u, v, p, current_time + dt, Nx, Ny, dx, dy);
    }
    
    current_time += dt;
}

// ==================================================================================
// Divergence Norm
// ==================================================================================
Real NavierStokes::computeDivergenceNorm() {
    Real div_norm = 0.0;
    #pragma omp parallel for reduction(+:div_norm)
    for (int j = 1; j < Ny; ++j) {
        for (int i = 1; i < Nx; ++i) {
            Real div_ij = divergence(i, j);
            div_norm += div_ij * div_ij;
        }
    }
    return std::sqrt(div_norm / ((Nx - 1) * (Ny - 1)));
}

// ==================================================================================
// Main Solve Driver
// ==================================================================================
void NavierStokes::solve(Real t_final, Real dt, int output_interval) {
    int num_steps = static_cast<int>(std::ceil(t_final / dt));
    
    std::cout << "\nStarting time integration...\n";
    std::cout << "  t_final = " << t_final << ", dt = " << dt << "\n";
    std::cout << "  Steps = " << num_steps << "\n\n";

    resetTimers();
    double t_total_start = omp_get_wtime();

    for (int step = 0; step <= num_steps; ++step) {
        if (step % output_interval == 0) {
            Real div_norm = computeDivergenceNorm();
            std::cout << "Step " << std::setw(5) << step 
                      << ", t = " << std::fixed << std::setprecision(4) << current_time
                      << ", |div·u| = " << std::scientific << std::setprecision(3) << div_norm 
                      << std::endl;
            
            char filename[64];
            snprintf(filename, sizeof(filename), "./results/output_%04d.vtk", step / output_interval);
            writeVTK(filename, current_time);
        }

        if (step < num_steps) {
            Real dt_actual = std::min(dt, t_final - current_time);
            if (dt_actual > 1e-14) {
                timeStepUpdate(dt_actual);
            }
        }
    }
    
    double t_total = omp_get_wtime() - t_total_start;
    std::cout << "\nTotal wall time: " << std::fixed << std::setprecision(3) 
              << t_total << "s\n";
    printTimingStats();
}

// ==================================================================================
// VTK Output
// ==================================================================================
void NavierStokes::writeVTK(const char* filename, Real time) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return;
    }
    
    file << "# vtk DataFile Version 3.0\n";
    file << "Navier-Stokes 2D, t = " << time << "\n";
    file << "ASCII\nDATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << (Nx + 1) << " " << (Ny + 1) << " 1\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING " << dx << " " << dy << " 1\n";
    file << "POINT_DATA " << (Nx + 1) * (Ny + 1) << "\n";

    file << "VECTORS velocity float\n";
    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            file << u[idx(i,j)] << " " << v[idx(i,j)] << " 0\n";
        }
    }

    file << "SCALARS pressure float 1\nLOOKUP_TABLE default\n";
    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            file << p[idx(i,j)] << "\n";
        }
    }

    file.close();
}