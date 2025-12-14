#include "TestCases.h"

namespace TestCases {

namespace Cavity {
void run(int N, Real t_final) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Real L = 1.0;
    Real Re = 100.0;
    Real U_lid = 1.0;
    Real nu = U_lid * L / Re;
    Real rho = 1.0;
    
    Real h = L / N;
    Real dt = std::min(0.001, 0.5 * (h*h) / (4*nu));

    if (rank == 0) {
        std::cout << "\n=== Lid-Driven Cavity (MPI) ===\n";
        std::cout << "Grid: " << (N+1) << " x " << (N+1) << "\n";
        std::cout << "dt = " << dt << "\n";
    }

    NavierStokes solver(N, N, L, L, rho, nu);

    solver.setInitialConditions([](Real, Real) { return 0.0; },
                                [](Real, Real) { return 0.0; },
                                [](Real, Real) { return 0.0; });

    solver.setBoundaryConditions([&](Real* u, Real* v, Real* p, Real t, int Nx, int Ny, Real dx, Real dy) {
        auto idx = [&](int i, int j) { return i + j * (Nx + 1); };
        
        // 1. Left/Right Walls
        for (int j = 1; j <= solver.ny_local; ++j) {
            u[idx(0, j)] = 0.0; v[idx(0, j)] = 0.0;
            u[idx(Nx, j)] = 0.0; v[idx(Nx, j)] = 0.0;
        }

        // 2. Bottom Wall (Rank 0)
        if (solver.rank == 0) {
            for (int i = 0; i <= Nx; ++i) {
                u[idx(i, 1)] = 0.0; v[idx(i, 1)] = 0.0; // Wall
                u[idx(i, 0)] = 0.0; v[idx(i, 0)] = 0.0; // Ghost
            }
        }

        // 3. Top Wall (Last Rank)
        if (solver.rank == solver.size - 1) {
            for (int i = 0; i <= Nx; ++i) {
                u[idx(i, solver.ny_local)] = 1.0; // Lid
                v[idx(i, solver.ny_local)] = 0.0;
                u[idx(i, solver.ny_local+1)] = 1.0; // Ghost
                v[idx(i, solver.ny_local+1)] = 0.0;
            }
        }
    });

    solver.solve(t_final, dt, 10);
}
} // namespace Cavity

namespace Poiseuille {

const Real Lx = 4.0;
const Real Ly = 1.0;
const Real U_max = 1.0;
const Real nu = 0.01;
const Real rho = 1.0;
const Real dpdx = -8.0 * nu * U_max / (Ly * Ly);  // = -0.08

// Exact solution
Real u_exact(Real x, Real y) {
    (void)x;
    return 4.0 * U_max * y * (Ly - y) / (Ly * Ly);
}

Real v_exact(Real x, Real y) {
    (void)x; (void)y;
    return 0.0;
}

void source_term(Real x, Real y, Real t, Real nu_loc, Real rho_loc, Real& fx, Real& fy) {
    (void)x; (void)y; (void)t; (void)nu_loc;
    fx = -dpdx / rho_loc;  // Body force drives the flow
    fy = 0.0;
}

void run(int N, Real t_final) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int Nx = 4 * N; 
    int Ny = N;
    Real h = Ly / Ny;
    Real dt = 0.9 * h * h / (4 * nu);

    if (rank == 0) {
        std::cout << "\n=== Poiseuille Flow Development (MPI) ===\n";
        std::cout << "Grid: " << (Nx+1) << " x " << (Ny+1) << "\n";
        std::cout << "dt = " << dt << "\n";
        std::cout << "Diffusion time scale: " << Ly*Ly/nu << "s\n";
    }

    NavierStokes solver(Nx, Ny, Lx, Ly, rho, nu);

    // Start from REST
    solver.setInitialConditions(
        [](Real, Real) { return 0.0; },
        [](Real, Real) { return 0.0; },
        [](Real, Real) { return 0.0; }
    );
    
    solver.setSourceTerm(source_term);

    // Neumann BCs at inlet/outlet, no-slip at walls
    solver.setBoundaryConditions([&](Real* u, Real* v, Real* p, Real t, 
                                      int Nx, int Ny, Real dx, Real dy) {
        (void)t; (void)p; (void)dy;
        auto idx = [&](int i, int j) { return i + j * (Nx + 1); };

        // Inlet (left): zero-gradient du/dx = 0
        for (int j = 1; j <= solver.ny_local; ++j) {
            u[idx(0, j)] = u[idx(1, j)];
            v[idx(0, j)] = 0.0;
        }
        
        // Outlet (right): zero-gradient du/dx = 0
        for (int j = 1; j <= solver.ny_local; ++j) {
            u[idx(Nx, j)] = u[idx(Nx-1, j)];
            v[idx(Nx, j)] = 0.0;
        }

        // Bottom wall (only rank 0)
        if (solver.s_neighbor == MPI_PROC_NULL) {
            for (int i = 0; i <= Nx; ++i) { 
                u[idx(i, 1)] = 0.0;
                v[idx(i, 1)] = 0.0;
                u[idx(i, 0)] = -u[idx(i, 2)];
                v[idx(i, 0)] = -v[idx(i, 2)];
            }
        }
        
        // Top wall (only last rank)
        if (solver.n_neighbor == MPI_PROC_NULL) {
            for (int i = 0; i <= Nx; ++i) {
                u[idx(i, solver.ny_local)] = 0.0;
                v[idx(i, solver.ny_local)] = 0.0;
                u[idx(i, solver.ny_local + 1)] = -u[idx(i, solver.ny_local - 1)];
                v[idx(i, solver.ny_local + 1)] = -v[idx(i, solver.ny_local - 1)];
            }
        }
    });

    solver.solve(t_final, dt, 20);
    
    // ============================================================
    // Compute errors against exact solution
    // ============================================================
    
    // Interior bounds (exclude physical boundaries)
    int j_lo = (solver.s_neighbor == MPI_PROC_NULL) ? 2 : 1;
    int j_hi = (solver.n_neighbor == MPI_PROC_NULL) ? solver.ny_local - 1 : solver.ny_local;
    
    Real u_err_local = 0.0, v_err_local = 0.0;
    Real u_norm_local = 0.0, v_norm_local = 0.0;
    Real u_max_err_local = 0.0, v_max_err_local = 0.0;
    int count_local = 0;
    
    for (int j = j_lo; j <= j_hi; ++j) {
        for (int i = 1; i < Nx; ++i) {
            Real x_pos = solver.x[i];
            Real y_pos = solver.y_local[j];
            int index = solver.idx(i, j);
            
            Real u_ex = u_exact(x_pos, y_pos);
            Real v_ex = v_exact(x_pos, y_pos);
            
            Real err_u = solver.u[index] - u_ex;
            Real err_v = solver.v[index] - v_ex;
            
            u_err_local += err_u * err_u;
            v_err_local += err_v * err_v;
            
            u_norm_local += u_ex * u_ex;
            v_norm_local += v_ex * v_ex;
            
            u_max_err_local = std::max(u_max_err_local, std::abs(err_u));
            v_max_err_local = std::max(v_max_err_local, std::abs(err_v));
            
            count_local++;
        }
    }
    
    // Global reductions
    Real u_err_global, v_err_global, u_norm_global, v_norm_global;
    Real u_max_err_global, v_max_err_global;
    int count_global;
    
    MPI_Allreduce(&u_err_local, &u_err_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&v_err_local, &v_err_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&u_norm_local, &u_norm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&v_norm_local, &v_norm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&u_max_err_local, &u_max_err_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&v_max_err_local, &v_max_err_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&count_local, &count_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n=== Errors vs Exact Poiseuille Solution ===\n";
        std::cout << std::scientific << std::setprecision(4);
        std::cout << "Relative L2(u) = " << std::sqrt(u_err_global / u_norm_global) << "\n";
        std::cout << "Absolute L2(v) = " << std::sqrt(v_err_global / count_global) << "\n";
        std::cout << "Max |u - u_exact| = " << u_max_err_global << "\n";
        std::cout << "Max |v - v_exact| = " << v_max_err_global << "\n";
    }
    
    // ============================================================
    // Print centerline velocity profile (rank 0 gathers data)
    // ============================================================
    
    // Gather centerline data from all ranks
    int i_center = Nx / 2;
    
    // Each rank sends its local centerline data
    std::vector<Real> local_y, local_u, local_u_exact;
    
    for (int j = j_lo; j <= j_hi; ++j) {
        Real y_pos = solver.y_local[j];
        Real u_val = solver.u[solver.idx(i_center, j)];
        Real u_ex = u_exact(0.0, y_pos);
        
        local_y.push_back(y_pos);
        local_u.push_back(u_val);
        local_u_exact.push_back(u_ex);
    }
    
    int local_count_profile = local_y.size();
    
    // Gather counts from all ranks
    std::vector<int> all_counts(size);
    MPI_Gather(&local_count_profile, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Compute displacements for MPI_Gatherv
    std::vector<int> displs(size);
    int total_count = 0;
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            displs[r] = total_count;
            total_count += all_counts[r];
        }
    }
    MPI_Bcast(&total_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Gather all data to rank 0
    std::vector<Real> all_y(total_count), all_u(total_count), all_u_exact(total_count);
    
    MPI_Gatherv(local_y.data(), local_count_profile, MPI_DOUBLE,
                all_y.data(), all_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_u.data(), local_count_profile, MPI_DOUBLE,
                all_u.data(), all_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_u_exact.data(), local_count_profile, MPI_DOUBLE,
                all_u_exact.data(), all_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        // Sort by y coordinate and print every ~10 points
        std::vector<std::tuple<Real, Real, Real>> profile_data;
        for (int i = 0; i < total_count; ++i) {
            profile_data.push_back({all_y[i], all_u[i], all_u_exact[i]});
        }
        std::sort(profile_data.begin(), profile_data.end());
        
        std::cout << "\n=== Centerline u-velocity (x = Lx/2) ===\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "    y        u_num      u_exact     error\n";
        
        int step = std::max(1, (int)profile_data.size() / 10);
        for (size_t i = 0; i < profile_data.size(); i += step) {
            Real y_val = std::get<0>(profile_data[i]);
            Real u_val = std::get<1>(profile_data[i]);
            Real u_ex = std::get<2>(profile_data[i]);
            std::cout << y_val << "    " << u_val << "    " << u_ex 
                      << std::scientific << std::setprecision(6) << std::get<1>(profile_data[i]) - u_ex << "\n";
            std::cout << std::fixed << std::setprecision(6);
        }
    }
}


} // namespace Poiseuille

}