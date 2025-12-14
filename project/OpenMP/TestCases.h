/**
 * @file TestCases.h
 * @brief Test cases for Navier-Stokes solver
 */
#ifndef TESTCASES_H
#define TESTCASES_H

#include "navierstokes.h"

namespace TestCases {

namespace Cavity {
    void run(int N, Real t_final);
}

namespace MMS {
    void run(int N, Real t_final);
}

namespace Poiseuille {
    void run(int N, Real t_final);
}

} // namespace TestCases

#endif
