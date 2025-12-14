/**
 * @file navierstokes.C
 * @brief 2D Incompressible Navier-Stokes Solver (MPI)
 */
#include "navierstokes.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>

// ==================================================================================
// Constructor
// ==================================================================================
NavierStokes::NavierStokes(int Nx, int Ny, Real Lx, Real Ly, Real rho, Real nu)
    : Nx(Nx), Ny(Ny), Lx(Lx), Ly(Ly), rho(rho), nu(nu), current_time(0.0),
      time_pressure(0.0), time_velocity(0.0), time_rhs(0.0), pressure_iterations(0)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    dx = Lx / Nx;
    dy = Ly / Ny;

    // Domain decomposition
    int N_nodes = Ny + 1;
    int rows_per_rank = N_nodes / size;
    int remainder = N_nodes % size;

    if (rank < remainder) 
    {
        ny_local = rows_per_rank + 1;
        j_start = rank * ny_local;
    } 
    else 
    {
        ny_local = rows_per_rank;
        j_start = rank * ny_local + remainder;
    }
    j_end = j_start + ny_local; 
    
    n_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    s_neighbor = (rank == 0)        ? MPI_PROC_NULL : rank - 1;

    // ALLOCATION
    // We need indices j=0 (Ghost South) to j=ny_local+1 (Ghost North).
    // Total rows needed = ny_local + 2.
    // We add +2 EXTRA padding rows at the end just to be safe against off-by-one errors.
    int allocated_rows = ny_local + 4; 
    int total_nodes = (Nx + 1) * allocated_rows;

    u = new Real[total_nodes]();
    v = new Real[total_nodes]();
    p = new Real[total_nodes]();
    u_old = new Real[total_nodes]();
    v_old = new Real[total_nodes]();
    rhs_u = new Real[total_nodes]();
    rhs_v = new Real[total_nodes]();
    rhs_p = new Real[total_nodes]();

    x = new Real[Nx + 1];
    for (int i = 0; i <= Nx; ++i) x[i] = i * dx;

    // y_local needs to support j=0..ny_local+1
    y_local = new Real[allocated_rows];
    for (int j = 0; j <= ny_local + 1; ++j) 
    {
        int global_j = j_start + (j - 1); 
        y_local[j] = global_j * dy; 
    }
    
    if (rank == 0) 
    {
        std::cout << "NavierStokes2D (MPI) initialized:\n";
        std::cout << "  Global Grid: " << (Nx+1) << " x " << (Ny+1) << "\n";
        std::cout << "  MPI Ranks: " << size << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
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
    delete[] y_local;
}

// ==================================================================================
// Timing functions
// ==================================================================================
void NavierStokes::resetTimers() {
    time_pressure = 0.0; time_velocity = 0.0; time_rhs = 0.0; pressure_iterations = 0;
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
    for (int j = 0; j <= ny_local; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            int index = idx(i, j);
            u[index] = u_init(x[i], y_local[j]);
            v[index] = v_init(x[i], y_local[j]);
            p[index] = p_init(x[i], y_local[j]);
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
    // Left/Right walls
    for (int j = 1; j <= ny_local; ++j) {
        p[idx(0, j)]  = p[idx(1, j)];
        p[idx(Nx, j)] = p[idx(Nx-1, j)];
    }
    
    // South ghost - ONLY rank 0 (physical boundary)
    if (s_neighbor == MPI_PROC_NULL) {
        for (int i = 0; i <= Nx; ++i) {
            p[idx(i, 1)] = p[idx(i, 2)];  // Wall = first interior (Neumann)
            p[idx(i, 0)] = p[idx(i, 1)];  // Ghost = wall
        }
    }
    // Otherwise, row 0 gets its value from halo exchange (don't touch it!)
    
    // North ghost - ONLY last rank (physical boundary)
    if (n_neighbor == MPI_PROC_NULL) {
        for (int i = 0; i <= Nx; ++i) {
            p[idx(i, ny_local)] = p[idx(i, ny_local - 1)];  // Wall = last interior
            p[idx(i, ny_local + 1)] = p[idx(i, ny_local)];  // Ghost = wall
        }
    }
//     p[idx(0, 0)]    = 0.5 * (p[idx(1, 0)]    + p[idx(0, 1)]);
//     p[idx(Nx, 0)]   = 0.5 * (p[idx(Nx-1, 0)] + p[idx(Nx, 1)]);
//     p[idx(0, ny_local)]   = 0.5 * (p[idx(1, ny_local)]   + p[idx(0, ny_local-1)]);
//     p[idx(Nx, ny_local)] = 0.5 * (p[idx(Nx-1, ny_local)] + p[idx(Nx, ny_local-1)]);
}

// ==================================================================================
// Pressure Poisson Solver (Red-Black Gauss-Seidel with SOR using MPI)
// ==================================================================================
void NavierStokes::solvePressurePoissonGS(int max_iter, Real tol_abs) {
    Real ax = 1.0 / (dx * dx);
    Real ay = 1.0 / (dy * dy);
    Real ap = -2.0 * (ax + ay);
    Real inv_ap = 1.0 / ap;
    
    // SOR relaxation factor
    const Real omega = 1.99;

    // Determine local j range: exclude physical boundaries
    int j_lo = 1;
    int j_hi = ny_local;
    
    if (s_neighbor == MPI_PROC_NULL) j_lo = 2;  // Rank 0: skip j=1 (bottom wall)
    if (n_neighbor == MPI_PROC_NULL) j_hi = ny_local - 1;  // Last rank: skip top wall

    exchangeHalos(p);
    applyPressureBC();  // Set ghost values before starting
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // Red sweep - INTERIOR ONLY (with SOR)
        for (int j = j_lo; j <= j_hi; ++j) {
            int global_j = j_start + (j - 1);
            int i_start = 1 + (global_j % 2);
            for (int i = i_start; i < Nx; i += 2) {
                int index = idx(i, j);
                Real p_gs = (rhs_p[index] 
                            - ax * (p[idx(i+1,j)] + p[idx(i-1,j)])
                            - ay * (p[idx(i,j+1)] + p[idx(i,j-1)])) * inv_ap;
                p[index] = (1.0 - omega) * p[index] + omega * p_gs;
            }
        }
        exchangeHalos(p);
        applyPressureBC();
        
        // Black sweep - INTERIOR ONLY (with SOR)
        for (int j = j_lo; j <= j_hi; ++j) {
            int global_j = j_start + (j - 1);
            int i_start = 1 + ((global_j + 1) % 2);
            for (int i = i_start; i < Nx; i += 2) {
                int index = idx(i, j);
                Real p_gs = (rhs_p[index] 
                            - ax * (p[idx(i+1,j)] + p[idx(i-1,j)])
                            - ay * (p[idx(i,j+1)] + p[idx(i,j-1)])) * inv_ap;
                p[index] = (1.0 - omega) * p[index] + omega * p_gs;
            }
        }
        exchangeHalos(p);
        applyPressureBC();
        
        // Convergence check - also only at interior
        if (iter % 100 == 0) {
            Real local_max_res = 0.0;
            for (int j = j_lo; j <= j_hi; ++j) {
                for (int i = 1; i < Nx; ++i) {
                    Real lap = ax * (p[idx(i+1,j)] - 2.0*p[idx(i,j)] + p[idx(i-1,j)]) +
                               ay * (p[idx(i,j+1)] - 2.0*p[idx(i,j)] + p[idx(i,j-1)]);
                    local_max_res = std::max(local_max_res, std::abs(lap - rhs_p[idx(i,j)]));
                }
            }
            Real global_max_res;
            MPI_Allreduce(&local_max_res, &global_max_res, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            
            // // for debugging
            // if (rank == 0 && current_time < 0.01 && iter % 1000 == 0) {
            //     std::cout << "  Pressure iter " << iter << ", residual = " << global_max_res << "\n";
            // }
            
            if (global_max_res < tol_abs) {
                pressure_iterations += iter + 1;
                return;
            }
        }
    }
    pressure_iterations += max_iter;
}

// ==================================================================================
// Divergence Norm
// ==================================================================================
Real NavierStokes::computeDivergenceNorm() {
    int j_lo = (s_neighbor == MPI_PROC_NULL) ? 2 : 1;
    int j_hi = (n_neighbor == MPI_PROC_NULL) ? ny_local - 1 : ny_local;
    
    Real local_sum = 0.0;
    int local_count = 0;
    for (int j = j_lo; j <= j_hi; ++j) {
        for (int i = 1; i < Nx; ++i) {
            Real div = divergence(i, j);
            local_sum += div * div;
            local_count++;
        }
    }
    
    Real global_sum = 0.0;
    int global_count = 0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    return std::sqrt(global_sum / global_count);
}

// ==================================================================================
// Halo Exchange (Non-Blocking)
// ==================================================================================
void NavierStokes::exchangeHalos(Real* data) {
    int count = Nx + 1;
    MPI_Request reqs[4];
    MPI_Status stats[4];
    int nreq = 0;

    // Pointers to data
    Real* send_north = &data[idx(0, ny_local)];   // My Top Real Row
    Real* recv_north = &data[idx(0, ny_local+1)]; // My Top Ghost Row
    Real* send_south = &data[idx(0, 1)];          // My Bottom Real Row
    Real* recv_south = &data[idx(0, 0)];          // My Bottom Ghost Row

    // 1. Post Receives First 
    // Recv from North (expecting tag 1 from their South send)
    if (n_neighbor != MPI_PROC_NULL) {
        MPI_Irecv(recv_north, count, MPI_DOUBLE, n_neighbor, 1, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    // Recv from South (expecting tag 0 from their North send)
    if (s_neighbor != MPI_PROC_NULL) {
        MPI_Irecv(recv_south, count, MPI_DOUBLE, s_neighbor, 0, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    // 2. Post Sends
    // Send to North (Tag 0)
    if (n_neighbor != MPI_PROC_NULL) {
        MPI_Isend(send_north, count, MPI_DOUBLE, n_neighbor, 0, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    // Send to South (Tag 1)
    if (s_neighbor != MPI_PROC_NULL) {
        MPI_Isend(send_south, count, MPI_DOUBLE, s_neighbor, 1, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    // 3. Wait for all
    MPI_Waitall(nreq, reqs, stats);
}

// ==================================================================================
// Time Step Update
// ==================================================================================
void NavierStokes::timeStepUpdate(Real dt) {
    double t_start = MPI_Wtime();

    exchangeHalos(u);
    exchangeHalos(v);

    // Copy OLD values. Size is whole block to be safe.
    int allocated_rows = ny_local + 4;
    size_t bytes = sizeof(Real) * (Nx + 1) * allocated_rows;
    std::memcpy(u_old, u, bytes);
    std::memcpy(v_old, v, bytes);
    
    // Bounds: exclude physical boundary rows
    int j_lo = (s_neighbor == MPI_PROC_NULL) ? 2 : 1;
    int j_hi = (n_neighbor == MPI_PROC_NULL) ? ny_local - 1 : ny_local;
    
    // Step 1: Intermediate Velocity - INTERIOR ONLY
    for (int j = j_lo; j <= j_hi; ++j) {
        for (int i = 1; i < Nx; ++i) {
            int index = idx(i, j);
            Real fx = 0.0, fy = 0.0;
            if (source_function) 
                source_function(x[i], y_local[j], current_time, nu, rho, fx, fy);
            
            rhs_u[index] = -convection_u(i,j) + nu * laplacian_u(i,j) + fx;
            rhs_v[index] = -convection_v(i,j) + nu * laplacian_v(i,j) + fy;
            
            u[index] = u_old[index] + dt * rhs_u[index];
            v[index] = v_old[index] + dt * rhs_v[index];
        }
    }
    
    time_rhs += MPI_Wtime() - t_start;

    if (bc_function) bc_function(u, v, p, current_time + dt, Nx, Ny, dx, dy);

    // Step 2: Pressure Setup
    t_start = MPI_Wtime();
    exchangeHalos(u);
    exchangeHalos(v);

    Real local_rhs_sum = 0.0;
    int local_count = 0;
    for (int j = j_lo; j <= j_hi; ++j) {
        for (int i = 1; i < Nx; ++i) {
            rhs_p[idx(i,j)] = (rho / dt) * divergence(i, j);
            local_rhs_sum += rhs_p[idx(i,j)];
            local_count++;
        }
    }

    Real global_rhs_sum;
    int global_count;
    MPI_Allreduce(&local_rhs_sum, &global_rhs_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    Real rhs_mean = global_rhs_sum / global_count;

    for (int j = j_lo; j <= j_hi; ++j) {
        for (int i = 1; i < Nx; ++i) {
            rhs_p[idx(i,j)] -= rhs_mean;
        }
    }
    time_rhs += MPI_Wtime() - t_start;

    // Pressure Solve
    t_start = MPI_Wtime();
    solvePressurePoissonGS(100000, 1e-10);
    time_pressure += MPI_Wtime() - t_start;

    // Subtract Mean Pressure (solvability condition for Neumann BC)
    Real local_p_sum = 0.0;
    for (int j = 1; j <= ny_local; ++j) {
        for (int i = 1; i < Nx; ++i) local_p_sum += p[idx(i,j)];
    }
    Real global_p_sum;
    MPI_Allreduce(&local_p_sum, &global_p_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Real p_mean = global_p_sum / ((Nx + 1) * (Ny + 1));
    
    for (int j = 1; j <= ny_local; ++j) {
        for (int i = 0; i <= Nx; ++i) p[idx(i,j)] -= p_mean;
    }

    // Step 3: Projection
    t_start = MPI_Wtime();
    exchangeHalos(p);
    
    for (int j = j_lo; j <= j_hi; ++j) {
        for (int i = 1; i < Nx; ++i) {
            int index = idx(i, j);
            u[index] -= (dt / rho) * dp_dx(i, j);
            v[index] -= (dt / rho) * dp_dy(i, j);
        }
    }
    
    if (bc_function) bc_function(u, v, p, current_time + dt, Nx, Ny, dx, dy);

    // In timeStepUpdate, after BCs are applied:
    if (current_time < 0.01) {
        // Check solution at rank boundaries
        int j_top = ny_local;      // Last interior row
        int j_bot = 1;             // First interior row
        int i_mid = Nx / 2;
        
        Real y_top = y_local[j_top];
        Real y_bot = y_local[j_bot];
        Real u_exact_top = 4.0 * 1.0 * y_top * (1.0 - y_top); 
        Real u_exact_bot = 4.0 * 1.0 * y_bot * (1.0 - y_bot);
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    current_time += dt;
}

// ==================================================================================
// Main Solve Driver
// ==================================================================================
void NavierStokes::solve(Real t_final, Real dt, int output_interval) {
    int num_steps = static_cast<int>(std::ceil(t_final / dt));
    
    if (rank == 0) {
        std::cout << "\nStarting time integration...\n";
        std::cout << "  t_final = " << t_final << ", dt = " << dt << "\n";
        std::cout << "  Steps = " << num_steps << "\n\n";
        resetTimers();
    }
    
    double t_total_start = MPI_Wtime();

    for (int step = 0; step <= num_steps; ++step) {
        if (step % output_interval == 0) {
            Real div_norm = computeDivergenceNorm();
            
            if (rank == 0) {
                std::cout << "Step " << std::setw(5) << step 
                          << ", t = " << std::fixed << std::setprecision(4) << current_time
                          << ", |div u| = " << std::scientific << std::setprecision(3) << div_norm 
                          << std::endl;
            }
            
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
    
    double t_total = MPI_Wtime() - t_total_start;
    if (rank == 0) {
        std::cout << "\nTotal wall time: " << std::fixed << std::setprecision(3) 
                  << t_total << "s\n";
        printTimingStats();
    }
}

// ==================================================================================
// Write VTK (Rank Distributed) (is not working like expected: need to FIX)
// ==================================================================================
void NavierStokes::writeVTK(const char* base_filename, Real time) {
    std::string s(base_filename);
    std::string rawname = s.substr(0, s.find_last_of(".")); 
    char rank_filename[256];
    snprintf(rank_filename, sizeof(rank_filename), "%s_rank%d.vtk", rawname.c_str(), rank);

    std::ofstream file(rank_filename);
    if (!file.is_open()) return;

    file << "# vtk DataFile Version 3.0\n";
    file << "NS Rank " << rank << "\n";
    file << "ASCII\nDATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << (Nx + 1) << " " << ny_local << " 1\n"; 
    file << "ORIGIN 0 " << y_local[1] << " 0\n"; 
    file << "SPACING " << dx << " " << dy << " 1\n";
    file << "POINT_DATA " << (Nx + 1) * ny_local << "\n";

    file << "VECTORS velocity float\n";
    for (int j = 1; j <= ny_local; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            file << u[idx(i, j)] << " " << v[idx(i, j)] << " 0\n";
        }
    }

    file << "SCALARS pressure float 1\nLOOKUP_TABLE default\n";
    for (int j = 1; j <= ny_local; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            file << p[idx(i, j)] << "\n";
        }
    }
    file.close();
}