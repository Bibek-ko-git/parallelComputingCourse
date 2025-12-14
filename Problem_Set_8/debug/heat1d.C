// Solve the heat equation in one space dimension - MPI Version

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <mpi.h>
#include "getLocalIndexBounds.h"

// Define a new type, "Real", which is equivalent to "double"

typedef double Real;

#include <string>
using std::string;
using std::max;

#include <ctime>

// -----------------------------------------------------------------------
//Return the current wall clock time in seconds
// -----------------------------------------------------------------------
inline double getCPU()
{
    return MPI_Wtime(); // wall-clock time for parallel codes
}


// -----------------------------------------------------------------------
// Function to save a vector into a matlab file.
// matlabFile (input) : save the vector in this file
// u_p        (input) : array of vector values
// name       (input) : name of the array
// (nd1a:nd1b) (input) : array dimensions
// -----------------------------------------------------------------------

int writeMatlabVector(FILE *matlabFile, Real *u_p, const char *name,
                      int nd1a, int nd1b)
{
    #define u(i) u_p[i-nd1a]

    const int numPerLine=8;  // number of entries per line
    // Save the vector as:
    // name = [ num num num ... num num num ];
    fprintf(matlabFile, "%s = [ ", name);
    for (int i=nd1a; i<=nd1b; i++) {
        fprintf(matlabFile, "%20.15e ", u(i));
        if( (i-nd1a)%numPerLine == numPerLine-1 ) 
        fprintf(matlabFile, "...\n"); // continuation line
    }
    fprintf(matlabFile, "];\n");
    return 0;
}


// -----------------------------------------------------------------------
// Parse command line arguments
// -----------------------------------------------------------------------
//int myRank; // global variable for processor rank
int parseCommand(int argc, char *argv[], int &nx, Real &tFinal, int &saveMatlab, 
                 int &debug, string &matlabFileName, int &commOption , int myRank)
{
    for (int i=1; i<argc; i++)
    {
        string arg = argv[i];
        if (arg.find("-nx=") == 0) {
            nx = atoi(arg.substr(4).c_str());
            if (myRank == 0)
                printf("parseCommand: SETTING -nx=%d\n", nx);
        }
        else if (arg.find("-tFinal=") == 0) {
            tFinal = atof(arg.substr(8).c_str());
            if (myRank == 0)
                printf("parseCommand: SETTING -tFinal=%e\n", tFinal);
        }
        else if (arg.find("-saveMatlab=") == 0) {
            saveMatlab = atoi(arg.substr(12).c_str());
            if (myRank == 0)
                printf("parseCommand: SETTING -saveMatlab=%d\n", saveMatlab);
        }
        else if (arg.find("-debug=") == 0) {
            debug = atoi(arg.substr(7).c_str());
            if (myRank == 0)
                printf("parseCommand: SETTING -debug=%d\n", debug);
        }
        else if (arg.find("-matlabFileName=") == 0) {
            matlabFileName = arg.substr(16);
            if (myRank == 0)
                printf("parseCommand: SETTING -matlabFileName=%s\n", matlabFileName.c_str());
        }
        else if (arg.find("-commOption=") == 0) {
            commOption = atoi(arg.substr(12).c_str());
            if (myRank == 0)
                printf("parseCommand: SETTING -commOption=%d\n", commOption);
        }
    }
    return 0;
}

// -----------------------------------------------------------------------

int main (int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    int myRank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (myRank == 0) {
        printf("Usage: mpirun -np heat1d -tFinal=f -nx=i -saveMatlab=[0|1|2] -debug=i -matlabFileName=s\n");
        printf("      nx = number of grid cells.\n");
        printf("      matlabFileName.m : save results to this file.\n");
    }
    

    #define TRIG_DD 1
    #define TRIG_NN 2
    #define POLY_DD 3
    #define POLY_NN 4

    // ==== Choose the solution here or compile with -DSOLUTION = [1|2|3|4] ====
    #ifndef SOLUTION
      //#define SOLUTION TRIG_DD
      //#define SOLUTION TRIG_NN
      #define SOLUTION POLY_DD
      //#define SOLUTION POLY_NN
    #endif
    // ==========================================================================

    const Real pi = M_PI;

    int debug=0; // set to 1 for debug info
    Real xa=0., xb=1.; // domain [xa,xb]
    Real kappa = .1; // heat conduction coefficient
    Real tFinal = .2;
    Real cfl = .9; // CFL number/ time step safety factor

    int nx =10; //default
    int saveMatlab = 0;
    int commOption = 0;  // 0=blocked, 1=non-blocked
    string matlabFileName="heat1d.m";


    parseCommand(argc, argv, nx, tFinal, saveMatlab, debug, matlabFileName, commOption, myRank);
    //create full grid

    //========== Grid and indexing (drawing)==================
    //                                       xa                    xb
    //                                   G---X---+---+---+ ... +---X---G
    //                                       0   1   2             Nx
    //                                       n1a                   n1b
    //                                   nd1a                      nd1b
    // C index                          0    1   2   3 ...

    Real dx = (xb - xa)/nx; // grid spacing
    const int numGhost = 1; // number of ghost cells
    const int n1a = 0; // first "interior" cell
    const int n1b = nx; // last "interior" cell
    const int nd1a = n1a - numGhost; // first cell, including ghost
    const int nd1b = n1b + numGhost; // last cell, including ghost
    const int nd1 = nd1b - nd1a + 1; // total number of grid points, including ghost

    // Get local index bounds
    int nx_l, n1a_l, n1b_l;
    getLocalIndexBounds(myRank, np, nx, nx_l, n1a_l, n1b_l);

    const int nd1a_l = n1a_l - numGhost;
    const int nd1b_l = n1b_l + numGhost;
    const int nd1_l = nd1b_l - nd1a_l + 1;

    // Open debug file for this processor
    FILE *debugFile = NULL;
    if (debug > 0) {
        char debugFileName[100];
        sprintf(debugFileName, "debug/heat1dNp%dProc%d.debug", np, myRank);
        debugFile = fopen(debugFileName, "w");
        fprintf(debugFile, "1\n\n");
    }


    // Create local grid
    Real *x_p = new Real[nd1_l];
    #define x(i) x_p[i-nd1a_l] // macro to access array

    for(int i=nd1a_l; i<=nd1b_l; i++) 
        x(i) = xa + (i-n1a)*dx;
    //if(debug > 1) {
        //for ( int i=nd1a; i<=nd1b; i++ ) 
      //      printf("x(%2d)=%12.4e\n",i,x(i));
    //}

    const int dirichlet =1, neumann=2;
    const int numberOfDimensions=1;
    int *boundaryCondition_p = new int [2*numberOfDimensions];
    #define boundaryCondition(side, axis) boundaryCondition_p[(side)+2*(axis)]

    const Real kx = 3.;
    const Real kxPi = kx*pi;
    const Real kappaPiSq = kappa*kxPi*kxPi;

    #if SOLUTION == TRIG_DD
        //True Solution for dirichlet BC's
        boundaryCondition(0,0) = dirichlet; // left boundary
        boundaryCondition(1,0) = dirichlet; // right boundary

        const char solutionName[] = "trueDD";

        #define UTRUE(x,t) sin(kxPi*(x))*exp(-kappaPiSq*(t))
        #define UTRUEX(x,t) kxPi*cos(kxPi*(x))*exp(-kappaPiSq*(t))
        #define FORCE(x,t) (0.)
    
    #elif SOLUTION == TRIG_NN
        //True Solution for neumann BC's
        boundaryCondition(0,0) = neumann; // left boundary
        boundaryCondition(1,0) = neumann; // right boundary 
        const char solutionName[] = "trueNN";

        #define UTRUE(x,t) cos(kxPi*(x))*exp(-kappaPiSq*(t))
        #define UTRUEX(x,t) -kxPi*sin(kxPi*(x))*exp(-kappaPiSq*(t))
        #define FORCE(x,t) (0.)

    #elif (SOLUTION == POLY_DD) || (SOLUTION == POLY_NN)
        //Polynomial manufactured solution
        
        #if SOLUTION == POLY_DD
            const char solutionName[] = "polyDD";
            boundaryCondition(0,0) = (myRank == 0) ? dirichlet : -2;
            boundaryCondition(1,0) = (myRank == np-1) ? dirichlet : -2;
        #else
            const char solutionName[] = "polyNN";
            boundaryCondition(0,0) = neumann; // left boundary
            boundaryCondition(1,0) = neumann; // right boundary
        #endif

        const Real b0=1., b1=.5, b2=.25; // coefficients of polynomial
        const Real a0=1., a1=.3;
        
        #define UTRUE(x,t) ( b0 + (x)*(b1 + (x)*b2))*(a0 + (t)*(a1))
        #define UTRUEX(x,t) (b1 + 2.*b2*(x))*(a0 + (t)*(a1))
        #define UTRUEXX(x,t) (2.*b2)*(a0 + (t)*(a1))
        #define UTRUET(x,t) (b0 + (x)*(b1 + (x)*b2))*(a1)

        //force = u_t - kappa*u_xx
        #define FORCE(x,t) (UTRUET(x,t) - kappa*UTRUEXX(x,t))

    #else
         printf("ERROR: unknown solution");
         abort();
    #endif  
    
    
    Real *u_p[2]; // two solution arrays to be used for current and new time level
    u_p[0] = new Real[nd1];
    u_p[1] = new Real[nd1];

    //Macros to define fortran like arrays
    #define uc(i) u_p[cur][i-nd1a_l] // current time level
    #define un(i) u_p[next][i-nd1a_l] // new time level

    //========== Initial conditions ==================
    Real t = 0.; // initial time
    int cur=0; //Current solution, index into u_p[]
    for (int i = nd1a_l; i <= nd1b_l; i++) 
        uc(i) = UTRUE(x(i), t);

    //if (debug > 0)
    //{
      //  printf("After initial conditions\n u=[");
       // for (int i = nd1a; i <= nd1b; i++)
        //    printf("%10.4e ", uc(i));
       // printf("]\n");
    //}

    // Time-step restriction is kappa*dt/dx^2 < .5
    const Real dx2      = dx*dx;
    Real dt             = cfl*.5*dx2/kappa; // time step
    const int numSteps  = ceil(tFinal/dt); // number of time steps
    dt                  = tFinal/numSteps; // adjust dt to reach the final time
    const Real rx       = kappa*dt/dx2; 


    if (myRank == 0) {
        printf("------------------- Solve the heat equation in 1D solution=%s ---------------------\n", solutionName);
        printf("np=%d, commOption=%d : %s\n", np, commOption, 
               commOption==0 ? "blockSendReceive" : "nonBlockSendReceive");
        printf("numGhost=%d, n1a=%d, n1b=%d, nd1a=%d, nd1b=%d\n", numGhost, n1a, n1b, nd1a, nd1b);
        printf("numSteps=%d, nx=%d, kappa=%g, tFinal=%g, boundaryCondition(0,0)=%d, boundaryCondition(1,0)=%d\n",
               numSteps, nx, kappa, tFinal, boundaryCondition(0,0), boundaryCondition(1,0));
    }



    if (debug > 0 && debugFile) {
        fprintf(debugFile, "------------------- Solve the heat equation in 1D solution=%s ---------------------\n", solutionName);
        fprintf(debugFile, "np=%d, commOption=%d : %s\n", np, commOption,
                commOption==0 ? "blockSendReceive" : "nonBlockSendReceive");
        fprintf(debugFile, "numGhost=%d, n1a=%d, n1b=%d, nd1a=%d, nd1b=%d\n", numGhost, n1a, n1b, nd1a, nd1b);
        fprintf(debugFile, "numSteps=%d, nx=%d, kappa=%g, tFinal=%g, boundaryCondition(0,0)=%d, boundaryCondition(1,0)=%d\n",
                numSteps, nx, kappa, tFinal, boundaryCondition(0,0), boundaryCondition(1,0));
        fprintf(debugFile, "myRank=%2d: nx=%3d, nx_l=%3d, [n1a_l,n1b_l]=[%3d,%3d], [nd1a_l,nd1b_l]=[%3d,%3d]\n",
                myRank, nx, nx_l, n1a_l, n1b_l, nd1a_l, nd1b_l);
        
        fprintf(debugFile, "x[%d:%d]=[", nd1a_l, nd1b_l);
        for (int i=nd1a_l; i<=nd1b_l; i++)
            fprintf(debugFile, "%.4f ", x(i));
        fprintf(debugFile, "]\n");
        
        fprintf(debugFile, "t=%.3e: u[%d:%d]=[", t, nd1a_l, nd1b_l);
        for (int i=nd1a_l; i<=nd1b_l; i++)
            fprintf(debugFile, "%.4f ", uc(i));
        fprintf(debugFile, "]\n");
    }

    //========== Time-stepping loop ==================

    Real cpu0 = getCPU(); // get initial CPU time

    for (int n = 0; n < numSteps; n++)
    {
        t = n*dt;

        const int cur = n % 2; // current solution, index into u_p[]
        const int next = (n + 1) % 2; // new solution, index into u_p[]

        // update the interior points
        for (int i = n1a_l; i <= n1b_l; i++)
        {
            un(i) = uc(i) + rx * (uc(i - 1) - 2. * uc(i) + uc(i + 1)) + dt*FORCE(x(i), t);
        } 

        // Exchange ghost points with neighbors
        if (commOption == 0) {
            // Blocked send/receive
            if (myRank > 0) {
                MPI_Send(&un(n1a_l), 1, MPI_DOUBLE, myRank-1, 0, MPI_COMM_WORLD);
                MPI_Recv(&un(n1a_l-1), 1, MPI_DOUBLE, myRank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (myRank < np-1) {
                MPI_Send(&un(n1b_l), 1, MPI_DOUBLE, myRank+1, 1, MPI_COMM_WORLD);
                MPI_Recv(&un(n1b_l+1), 1, MPI_DOUBLE, myRank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        else if (commOption == 1) {
            // Non-blocking send/receive
            MPI_Request requests[4];
            int numRequests = 0;
            
            // Post receives first
            if (myRank > 0) {
                MPI_Irecv(&un(n1a_l-1), 1, MPI_DOUBLE, myRank-1, 1, MPI_COMM_WORLD, &requests[numRequests++]);
            }
            if (myRank < np-1) {
                MPI_Irecv(&un(n1b_l+1), 1, MPI_DOUBLE, myRank+1, 0, MPI_COMM_WORLD, &requests[numRequests++]);
            }
            
            // Post sends
            if (myRank > 0) {
                MPI_Isend(&un(n1a_l), 1, MPI_DOUBLE, myRank-1, 0, MPI_COMM_WORLD, &requests[numRequests++]);
            }
            if (myRank < np-1) {
                MPI_Isend(&un(n1b_l), 1, MPI_DOUBLE, myRank+1, 1, MPI_COMM_WORLD, &requests[numRequests++]);
            }
            
            // Wait for all communications to complete
            MPI_Waitall(numRequests, requests, MPI_STATUSES_IGNORE);
        }

        // Physical boundary conditions
        if (myRank == 0) {
            const int i = n1a_l;
            un(i) = UTRUE(x(i), t+dt);
            un(i - 1) = 3.*un(i) - 3.*un(i + 1) + un(i + 2);
        }
        if (myRank == np-1) {
            const int i = n1b_l;
            un(i) = UTRUE(x(i), t+dt);
            un(i + 1) = 3.*un(i) - 3.*un(i - 1) + un(i - 2);
        }
            //needs to be revisted to make options for Neumann BCs on both ends
            //else
            //{
             //   // Neumann BC
               // un(i-is) = un(i+is) - 2.*is*dx*UTRUEX(x(i), t+dt);
            //}
        //} 

        if (debug > 1 && debugFile) {
            Real maxErr = 0.;
            for (int i = nd1a_l; i <= nd1b_l; i++) {
                Real err = fabs(un(i) - UTRUE(x(i), t + dt));
                maxErr = max(maxErr, err);
            }
            fprintf(debugFile, "Step %d: t=%9.3e, maxErr=%9.2e\n", n+1, t+dt, maxErr);
        }

    } // end time step loop

    Real cpuTimeStep = getCPU() - cpu0; // CPU time for time-stepping
    Real maxCpuTimeStep;
    MPI_Reduce(&cpuTimeStep, &maxCpuTimeStep, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // -------check the error -----------

    t += dt; // final time

    cur = numSteps % 2;
    Real maxErr_local = 0.;
    for (int i = nd1a_l; i <= nd1b_l; i++) {
        Real err = fabs(uc(i) - UTRUE(x(i), t));
        maxErr_local = max(maxErr_local, err);
    }
    
    Real maxErr_global;
    MPI_Reduce(&maxErr_local, &maxErr_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myRank == 0) {
        printf("np=%2d, numSteps=%3d, nx=%3d, maxErr=%9.2e, cpu=%9.2e(s)\n",
               np, numSteps, nx, maxErr_global, maxCpuTimeStep);
    }

    // Write debug file final info
    if (debug > 0 && debugFile) {
        fprintf(debugFile, "t=%.3e: u[%d:%d]=[", t, nd1a_l, nd1b_l);
        for (int i=nd1a_l; i<=nd1b_l; i++)
            fprintf(debugFile, "%.4f ", uc(i));
        fprintf(debugFile, "]\n");
        
        fprintf(debugFile, "t=%7.2e: error[%d:%d]=[", t, nd1a_l, nd1b_l);
        for (int i=nd1a_l; i<=nd1b_l; i++) {
            Real err = fabs(uc(i) - UTRUE(x(i), t));
            fprintf(debugFile, "%.2e ", err);
        }
        fprintf(debugFile, "]\n");
        fclose(debugFile);
    }

    // Gather solution to rank 0 for output
    if (saveMatlab > 0) {
        Real *u_global = NULL;
        Real *x_global = NULL;
        Real *error_global = NULL;
        
        if (myRank == 0) {
            const int nd1 = nd1b - nd1a + 1;
            u_global = new Real[nd1];
            x_global = new Real[nd1];
            error_global = new Real[nd1];
        }

        // Gather using blocked send/receive
        if (myRank == 0) {
            // Copy local data
            for (int i=nd1a_l; i<=nd1b_l; i++) {
                x_global[i-nd1a] = x(i);
                u_global[i-nd1a] = uc(i);
                error_global[i-nd1a] = uc(i) - UTRUE(x(i), t);
            }
            
            // Receive from other processors
            for (int p=1; p<np; p++) {
                int nx_p, n1a_p, n1b_p;
                getLocalIndexBounds(p, np, nx, nx_p, n1a_p, n1b_p);
                int nd1a_p = n1a_p - numGhost;
                int nd1b_p = n1b_p + numGhost;
                int nd1_p = nd1b_p - nd1a_p + 1;
                
                Real *u_temp = new Real[nd1_p];
                Real *x_temp = new Real[nd1_p];
                
                MPI_Recv(u_temp, nd1_p, MPI_DOUBLE, p, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(x_temp, nd1_p, MPI_DOUBLE, p, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                for (int i=nd1a_p; i<=nd1b_p; i++) {
                    x_global[i-nd1a] = x_temp[i-nd1a_p];
                    u_global[i-nd1a] = u_temp[i-nd1a_p];
                    error_global[i-nd1a] = u_temp[i-nd1a_p] - UTRUE(x_temp[i-nd1a_p], t);
                }
                
                delete [] u_temp;
                delete [] x_temp;
            }
            
            // Check error
            Real gatheredMaxErr = 0.;
            for (int i=nd1a; i<=nd1b; i++) {
                gatheredMaxErr = max(gatheredMaxErr, fabs(error_global[i-nd1a]));
            }
            printf("np=%d, myRank=%d, Gathered solution: maxErr=%9.2e\n", np, myRank, gatheredMaxErr);
            
            // Write matlab file
            FILE *matlabFile = fopen(matlabFileName.c_str(), "w");
            fprintf(matlabFile, "%% File written by heat1d.C\n");
            fprintf(matlabFile, "xa=%g; xb=%g; kappa=%g; t=%g; maxErr=%10.3e; cpuTimeStep=%10.3e; np=%d;\n",
                    xa, xb, kappa, t, gatheredMaxErr, maxCpuTimeStep, np);
            fprintf(matlabFile, "nx=%d; dx=%14.6e; numGhost=%d; n1a=%d; n1b=%d; nd1a=%d; nd1b=%d;\n",
                    nx, dx, numGhost, n1a, n1b, nd1a, nd1b);
            fprintf(matlabFile, "solutionName='%s';\n", solutionName);
            
            writeMatlabVector(matlabFile, x_global, "x", nd1a, nd1b);
            writeMatlabVector(matlabFile, u_global, "u", nd1a, nd1b);
            writeMatlabVector(matlabFile, error_global, "err", nd1a, nd1b);
            
            fclose(matlabFile);
            printf("Wrote file %s\n", matlabFileName.c_str());
            
            delete [] u_global;
            delete [] x_global;
            delete [] error_global;
        }
        else {
            // Send local data to rank 0
            MPI_Send(u_p[cur], nd1_l, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
            MPI_Send(x_p, nd1_l, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
        }
    }

    delete [] u_p[0];
    delete [] u_p[1];
    delete [] x_p;
    delete [] boundaryCondition_p;

    MPI_Finalize();
    return 0;

}