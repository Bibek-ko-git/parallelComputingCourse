#ifndef TRIDIAGONAL_H
#define TRIDIAGONAL_H

// to solve the tridiagonal systems

#include "A++.h"

typedef double Real;
typedef doubleSerialArray RealArray;

int factorTridiagonalMatrix( RealArray & Ax);

int solveTridiagonal( RealArray & Ax, RealArray & rhs);

#endif