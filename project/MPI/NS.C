/**
 * @file NS.C
 * @brief Main driver for Navier-Stokes solver (Projection Method, OpenMP)
 * 
 * Usage: ./navierstokes <test_case> [grid_size] [t_final]
 *   test_case: 1=MMS, 2=Poiseuille, 3=Cavity
 *   grid_size: N (default 64)
 *   t_final: final simulation time (default 1.0)
 */
#include "TestCases.h"
#include <iostream>
#include <cstdlib>
#include <mpi.h> // <--- Include MPI

int main(int argc, char** argv) {
    // 1. Initialize MPI BEFORE anything else
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Only Rank 0 prints the header
    if (rank == 0) {
        std::cout << "========================================\n";
        std::cout << "  Navier-Stokes Solver (MPI)\n";
        std::cout << "========================================\n";
    }
    
    int test_case = 3;  // Default: cavity
    int N = 64;
    Real t_final = 1.0;
    
    if (argc > 1) test_case = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) t_final = std::atof(argv[3]); // Use atof for float/double
    
    switch (test_case) {
        // case 1:
        //     TestCases::MMS::run(N, t_final);
        //     break;
        case 1:
            TestCases::Poiseuille::run(N, t_final);
            break;
        case 2:
            TestCases::Cavity::run(N, t_final);
            break;
        default:
            if (rank == 0) {
                std::cerr << "Unknown test case: " << test_case << "\n";
                std::cerr << "Usage: " << argv[0] << " <test_case> [s] [t_final]\n";
                std::cerr << " 1 = Poiseuille, 2 = Cavity\n";
            }
            // Don't return 1 here, finalize first or abort
            MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "  Done!\n";
        std::cout << "========================================\n";
    }
    
    // 2. Finalize MPI at the very end
    MPI_Finalize();
    return 0;
}