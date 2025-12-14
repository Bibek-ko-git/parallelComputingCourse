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
#include <omp.h>

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "  Navier-Stokes Solver (OpenMP)\n";
    std::cout << "========================================\n";
    
    int test_case = 3;      // Default: cavity
    int N = 64;             // Default grid size
    Real t_final = 1.0;     // Default final time
    
    if (argc > 1) test_case = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) t_final = std::atof(argv[3]);
    std::cout << "Threads: " << omp_get_max_threads() << "\n";
    
    
    switch (test_case) {
        case 1:
            TestCases::MMS::run(N, t_final);
            break;
        case 2:
            TestCases::Poiseuille::run(N, t_final);
            break;
        case 3:
            TestCases::Cavity::run(N, t_final);
            break;
        default:
            std::cerr << "Unknown test case: " << test_case << "\n";
            std::cerr << "Usage: " << argv[0] << " <test_case> [N] [t_final]\n";
            std::cerr << "  1 = MMS, 2 = Poiseuille, 3 = Cavity\n";
            return 1;
    }
    
    std::cout << "\n========================================\n";
    std::cout << "  Done!\n";
    std::cout << "========================================\n";
    
    return 0;
}
