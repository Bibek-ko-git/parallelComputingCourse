//
// solve the heat equation in 1D
//

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <unistd.h> 

#include <mpi.h>
#include "getLocalIndexBounds.h" // required to determine the distribution of the grid points
#include "parseCommand.h"      // required to parse command line arguments

// Let's define a new type "Real" number which is equivalent to double
typedef double Real;

#include <string>
using std::string;      // to avoid writing std::string everywhere
using std::max;         // to avoid ambiguity with std::max and fmax and just use max
using std::min;         // to avoid ambiguity with std::min and fmin and just use min

#include <ctime> 
//
// Return the current wall-clock time in seconds
//
inline double getCPU()
{
    return MPI_Wtime(); // use MPI timer

}
// =======================================================================
// Return the max values of a scalar over all processors in a communicator
// /processor: return the result to this processor (-1 equals all processors)
// =======================================================================
Real 
getMaxValue(Real value, int processor =-1, MPI_Comm comm = MPI_COMM_WORLD)
{
    Real maxValue=value;
    if (processor==-1)
        MPI_Allreduce(&value, &maxValue, 1, MPI_DOUBLE, MPI_MAX, comm);
    else
        MPI_Reduce(&value, &maxValue, 1 , MPI_DOUBLE, MPI_MAX, processor, comm);
    return maxValue;
}
// =======================================================================
// Return the min values of a scalar over all processors in a communicator
// /processor: return the result to this processor (-1 equals all processors)
// =======================================================================
Real 
getMinValue(Real value, int processor =-1, MPI_Comm comm = MPI_COMM_WORLD)
{
    Real minValue=value;
    if (processor==-1)
        MPI_Allreduce(&value, &minValue, 1, MPI_DOUBLE, MPI_MIN, comm);
    else
        MPI_Reduce(&value, &minValue, 1 , MPI_DOUBLE, MPI_MIN, processor, comm);
    return minValue;
}
// 
// Function to save a vector to a matlab file.
// matlabFile (input) : save vector to this file
// u_p        (input) : array of vector values
// name       (input) : name for array
// (nd1a:nd1b)(input) : array dimensions
//
int writeMatlabVector (FILE *matlabFile, Real *u_p, const char *name, int nd1a, int nd1b)
{
    #define u(i) u_p[i - nd1a]               // macro to access the vector elements

    const int numPerLine = 8;                // number of values per line
    // Save the vector as : 
    // name = [ num num num num num ...
    //          num num num num num ];
    fprintf(matlabFile, "%s=[ ",name);       // start of array
    for (int i = nd1a; i<= nd1b; i++)
    {
        fprintf (matlabFile, "%20.15e ", u(i));     // write the value
        if ((i - nd1a) % numPerLine == numPerLine -1)
            fprintf(matlabFile, "...\n");           // continuation line
    }
    fprintf(matlabFile,"];\n"); 

    #undef u
    return 0;
}

// To exchange the ghost point values between neighboring processors
void exchangeGhostPoints(Real *u_p, int n1a, int n1b, int nd1a, int myRank, int np, int commOption, MPI_Comm comm = MPI_COMM_WORLD)
{
    #define u(i) u_p[i-nd1a]

    if (commOption == 0) // Blocked Sending/Receiving
    {  
        MPI_Status status;

        // Exchange with left neighbor
        if (myRank > 0) 
        {
            MPI_Sendrecv(&u(n1a), 1, MPI_DOUBLE, myRank - 1, 1,      // send left
                        &u(n1a - 1), 1, MPI_DOUBLE, myRank - 1, 0,   // receive from left
                        comm, &status);
        }
        
        // Exchange with right neighbor
        if (myRank < np - 1) 
        {
            MPI_Sendrecv(&u(n1b), 1, MPI_DOUBLE, myRank + 1, 0,      // send right
                        &u(n1b + 1), 1, MPI_DOUBLE, myRank + 1, 1,   // receive from right
                        comm, &status);
        }
    }
    else 
    {
        MPI_Request requests[4];
        int numRequests =0;

        // The non-blocking receives first
        if (myRank > 0) // has left neighbor
        {
            MPI_Irecv( &u(n1a -1), 1, MPI_DOUBLE, myRank -1, 0, comm, &requests[numRequests++]); // receive left ghost point
        }
        if (myRank < np -1) // has right neighbor
        {
            MPI_Irecv( &u(n1b +1), 1, MPI_DOUBLE, myRank +1, 1, comm, &requests[numRequests++]); // receive right ghost point    
        }
        // Now the non-blocking sends
        if (myRank < np -1) // has right neighbor
        {
            MPI_Isend( &u(n1b), 1, MPI_DOUBLE, myRank +1, 0, comm, &requests[numRequests++]); // send rightmost real point
        }
        if (myRank > 0) // has left neighbor
        {
            MPI_Isend( &u(n1a), 1, MPI_DOUBLE, myRank -1, 1, comm, &requests[numRequests++]); // send leftmost real point
        }
        // Wait for all non-blocking operations to complete
        MPI_Waitall( numRequests, requests, MPI_STATUSES_IGNORE );
    }

    #undef u
}


int main (int argc, char* argv[])
{
    MPI_Init( &argc, &argv);               // initialize the MPI 
    int myRank;
    MPI_Comm_rank( MPI_COMM_WORLD, &myRank);    // the processor number
    int np;
    MPI_Comm_size( MPI_COMM_WORLD, &np);        // total number of processors

    if (myRank==0)
    {
        printf("Usage: heat1d -nx=i -debug=i -tFinal=f -commOption=i -matlabFineName=s]\n" //saves matlabfile if called
           "       nx  = number of grid cells.\n"
           "       matlabFileName.m : save results to this file.\n");
    }
    
    #define TRIG_DD 1 // for Trigonometric manufactured solution with Dirichlet-Dirichlet BCs
    #define TRIG_NN 2 // for Trigonometric manufactured solution with Neumann-Neumann BCs
    #define POLY_DD 3 // for Polynomial manufactured solution with Dirichlet-Dirichlet BCs
    #define POLY_NN 4 // for Polynomial manufactured solution with Neumann-Neumann BCs

    // ==== Choose the solution here or comple with -DSOLUTION=[1|2,3|4] to select at compile time ====
    #ifndef SOLUTION
        // #define SOLUTION TRIG_DD
        // #define SOLUTION TRIG_NN
        #define SOLUTION POLY_DD
        // #define SOLUTION POLY_NN
    #endif
    // =============================================================================================
    
    const Real pi = M_PI; // 3.14159265358979323846;

    int debug = 0;              // set 1 for debug 1 info

    Real xa = 0., xb = 1.; // domain [xa,xb]
    Real kappa = .1;    // diffusion coefficient
    Real tFinal = .2; // final time
    Real cfl = .9; // CFL number (time safety factor)

    int Nx = 10;   // default  
    int saveMatlab = 0; // save matlab file flag
    int commOption = 0; // communication option : 0=blocking send/receive and 1 = non-blocking
    string matlabFileName = "heat1d.m";
    
    for (int i = 1; i < argc; i++)
    {
        string line = argv[i];
        
        if (parseCommand(line, "-nx=", Nx, myRank == 0)) continue;
        if (parseCommand(line, "-Nx=", Nx, myRank == 0)) continue;
        if (parseCommand(line, "-debug=", debug, myRank == 0)) continue;
        if (parseCommand(line, "-tFinal=", tFinal, myRank == 0)) continue;
        if (parseCommand(line, "-commOption=", commOption, myRank == 0)) continue;
        if (parseCommand(line, "-matlabFileName=", matlabFileName, myRank == 0)) continue; // saves matlab file if called 
    }


    // ==== Setting up the grid ====
    //  xa                                     xb
    //  G---X---+---+---+---+-- ... ---+---X---G
    //      0   1   2                     Nx
    //     n1a                            n1b
    // nd1a                                   nd1b
    // C index: 0 1 2 3 ...
    // G = ghost cell

    Real dx = (xb - xa)/Nx; // grid spacing
    const int numghost = 1; // number of ghost cells
    // Updating the index boynds for a local system
    const int n1a = 0; // local first cell index
    const int n1b = Nx; // last cell index
    const int nd1a = n1a - numghost; // local first data index
    const int nd1b = n1b + numghost; // local last data index
    const int nd1 = nd1b - nd1a + 1; // total number of grid points 

    // Distributing the grid across the processors 
    int nx_l, n1a_l, n1b_l; // local values for this processor
    getLocalIndexBounds( myRank, np, Nx, nx_l, n1a_l, n1b_l ); // to determine the local bounds

    const int nd1a_l = n1a_l - numghost;
    const int nd1b_l = n1b_l + numghost;
    const int nd1_l = nd1b_l - nd1a_l + 1;

    // Create an array of grid points;
    Real *x_p = new Real[nd1_l];
    #define x(i) x_p[i - nd1a_l]

    for ( int i = nd1a_l; i <= nd1b_l; i++)
        x(i) = xa + (i)*dx; // cell center for global array 

    FILE *debugFile = NULL;

    const int dirichlet = 1, neumann = 2;
    const int numberOfDimensions = 1;
    int *boundaryCondition_p = new int [2*numberOfDimensions];          // array to hold boundary conditions
    #define boundaryCondition(side,axis) boundaryCondition_p[(side)+2*(axis)] // macro to access the array

    const Real kx = 3.;
    const Real kxPi = kx * pi;
    const Real kappaPiSq = kappa*kxPi*kxPi;

    #if SOLUTION == TRIG_DD
        // True solution for the Dirichlet BC's
        boundaryCondition(0,0) = (myRank == 0) ? dirichlet : -2; // only set BC on the leftmost processor
        boundaryCondition(1,0) = (myRank == np -1) ? dirichlet : -2; // only set BC on the rightmost processor

        const char solutionName[] = "trueDD";

        #define UTRUE(x,t) sin(kxPi*(x))*exp(-kappaPiSq*(t))
        #define UTRUEX(x,t) kxPi*cos(kxPi*(x))*exp(-kappaPiSq*(t))
        #define FORCE(x,t) (0.) // forcing term

    #elif SOLUTION == TRIG_NN
        // True solution for the Neumann BC's
        boundaryCondition(0,0) = (myRank == 0) ? neumann : -2; // only set BC on the leftmost processor
        boundaryCondition(1,0) = (myRank == np -1) ? neumann : -2; // only set BC on the rightmost processor

        const char solutionName[] = "trueNN";

        #define UTRUE(x,t) cos(kxPi*(x))*exp(-kappaPiSq*(t))
        #define UTRUEX(x,t) -kxPi*sin(kxPi*(x))*exp(-kappaPiSq*(t))
        #define FORCE(x,t) (0.) // forcing term

    #elif (SOLUTION == POLY_DD) || (SOLUTION == POLY_NN)
        //  True solution for the manufacture solution

        #if SOLUTION == POLY_DD
            const char solutionName[] = "polyDD";
            boundaryCondition(0,0) = (myRank == 0) ? dirichlet : -2; // only set BC on the leftmost processor
            boundaryCondition(1,0) = (myRank == np -1) ? dirichlet : -2; // only set BC on the rightmost processor

        #else 
            const char solutionName[] = "polyNN";
            boundaryCondition(0,0) = (myRank == 0) ? neumann : -2; // only set BC on the leftmost processor
            boundaryCondition(1,0) = (myRank == np -1) ? neumann : -2; // only set BC on the rightmost processor
        #endif

        const Real b0 = 1., b1 = .5, b2 = .25;
        const Real a0 = 1., a1 = .3; 

        #define UTRUE(x,t) (b0 + (x)*( b1 + (x)*b2 ))*( a0 + (t)*( a1 ) )
        #define UTRUEX(x,t) ( b1 + 2.*(x)*b2 )*( a0 + (t)*( a1 ) )
        #define UTRUET(x,t) (b0 + (x)*( b1 + (x)*b2 ))*( a1 )
        #define UTRUEXX(x,t) ( 2.*b2 )*( a0 + (t)*( a1 ) )

        // force = u_t - kappa*u_xx;
        #define FORCE(x,t) ( UTRUET(x,t) - kappa*UTRUEXX(x,t) )

    #else
        printf("ERROR: unknown solution");
        abort();
    #endif

    
    Real *u_p[2];               // two arrays will be used for the current and new times
    u_p[0] = new Real[nd1_l];    // allocate the arrays
    u_p[1] = new Real[nd1_l];

    // Macros to define fortran like arrays
    #define uc(i) u_p[cur][i - nd1a_l] // current time
    #define un(i) u_p[next][i - nd1a_l]

    // initial conditions
    Real t = 0.;
    int cur = 0; // "current" solution, index into u_p array
    for (int i = nd1a_l; i<=nd1b_l; i++)
        uc(i) =  UTRUE(x(i),t);

    // halo exchange for the *current* buffer
    //exchangeGhostPoints(u_p[cur], n1a, n1b, nd1a, myRank, np, commOption);

    if (debug > 0) 
    {   
        char debugFileName[256];
        sprintf(debugFileName,"debug/heat1dNp%dProc%d.debug",np,myRank);
        debugFile = fopen(debugFileName, "w");
        fprintf(debugFile, "------------------- Solve the heat equation in 1D solution=%s --------------------np=%d, commOption=%d : %s\n",
            solutionName, np, commOption, commOption == 0 ? "blockSendReceive" : "nonBlockSendReceive");
        fprintf(debugFile, "numGhost=%d, n1a=%d, n1b=%d, nd1a=%d, nd1b=%d\n", 
            numghost, 0, Nx, -numghost, Nx + numghost);
        fclose(debugFile);
    }

    // Time step restriction is kappa*dt/dx^2 <= 1/2
    const Real dx2     = dx*dx;
    Real dt            = cfl*.5*dx2/kappa; // time step "dt" adjusted below
    const int numSteps = ceil(tFinal/dt); 
    dt                 = tFinal/numSteps; // adjust dt so that we finish at tFinal exactly
    const Real rx      = kappa*dt/dx2; 
    
    // --------Time-Stepping loop -------------
    Real cpu0 = getCPU(); 
    if (debug > 0) 
    {
        char debugFileName[256];
        sprintf(debugFileName, "debug/heat1dNp%dProc%d.debug", np, myRank);
        FILE *debugFile = fopen(debugFileName, "a");  // append mode
        
        fprintf(debugFile, "numSteps=%d, nx=%d, kappa=%g, tFinal=%g, boundaryCondition(0,0)=%d, boundaryCondition(1,0)=%d\n",
                numSteps, Nx, kappa, tFinal, boundaryCondition(0,0), boundaryCondition(1,0));
        fprintf(debugFile, "myRank=%2d: nx=%3d, nx_l=%3d, [n1a_l,n1b_l]=[%3d,%3d], [nd1a_l,nd1b_l]=[%3d,%3d]\n",
                myRank, Nx, nx_l, n1a_l, n1b_l, nd1a_l, nd1b_l);
        
        // Write x values
        fprintf(debugFile, "x[%d:%d]=[", nd1a_l, nd1b_l);
        for (int i = nd1a_l; i <= nd1b_l; i++)
            fprintf(debugFile, "%.4f ", x(i));
        fprintf(debugFile, "]\n");
       

        // Write initial u values
        fprintf(debugFile, "t=%.3e: u[%d:%d]=[", t, nd1a_l, nd1b_l);
        for (int i = nd1a_l; i <= nd1b_l; i++)
            fprintf(debugFile, "%.4f ", uc(i));
        fprintf(debugFile, "]\n");

        fclose(debugFile);
    }

    for( int n=0; n<numSteps; n++)
    { 
        t = n*dt; // current time

        const int cur = n % 2; // current time level
        const int next = (n+1) % 2; // next time level
        
        // --- update the interior points ----
        for( int i=n1a_l; i<=n1b_l; i++ )
        {
            un(i) = uc(i) + rx*( uc(i+1) - 2.*uc(i) + uc(i-1) ) + dt*FORCE( x(i),t );
        }

        exchangeGhostPoints(u_p[next], n1a_l, n1b_l, nd1a_l, myRank, np, commOption);
        // ---- boundary condtitions only for the points in the local processor ----
        if (myRank == 0) 
        {
            if( boundaryCondition(0,0)==dirichlet )
            {
                const int i = n1a_l;
                un(i) = UTRUE(x(i), t + dt);
                un(i - 1) = 3.*un(i) - 3.*un(i + 1) + un(i + 2); // extrapolate ghost
            }
            else 
            {
                const int i = n1a_l;
                un(i - 1) = un(i + 1) - 2.*dx*UTRUEX(x(i), t + dt);
            }
        }
        if (myRank == np - 1)
        {
            if (boundaryCondition(1,0) == dirichlet)
            {
                const int i = n1b_l;
                un(i) = UTRUE(x(i), t + dt);
                un(i + 1) = 3.*un(i) - 3.*un(i - 1) + un(i - 2); // extrapolate ghost
            }
            else 
            {
                const int i = n1b_l;
                un(i + 1) = un(i - 1) + 2.*dx*UTRUEX(x(i), t + dt);
            }
        }       
        
    } // end time-stepping loop

    Real cpuTimeStep = getCPU()-cpu0;

    // ---- check the error -----
    t +=dt; // tFinal
    if( fabs(t-tFinal) > 1e-3*dt/tFinal )
    {
        printf("ERROR: AFTER TIME_STEPPING: t=%16.8e IS NOT EQUAL to tFinal=%16.8e\n",t,tFinal);
    }
    Real *error_p = new Real[nd1_l];
    #define error(i) error_p[i-nd1a_l]

    cur = numSteps % 2;
    Real maxErr=0.;
    for( int i=nd1a_l; i<=nd1b_l; i++ )
    {
        error(i) = uc(i) - UTRUE(x(i),t);
        maxErr = max( maxErr, fabs(error(i)) );
    }

    if( debug>0 )
        {
            // compute the error
            char debugFileName[256];
            sprintf(debugFileName, "debug/heat1dNp%dProc%d.debug", np, myRank);
            FILE *debugFile = fopen(debugFileName, "a");  // append mode
            
            // Write calculated u values
            fprintf(debugFile, "t=%.3e: u[%d:%d]=[", t, nd1a_l, nd1b_l);
            for (int i = nd1a_l; i <= nd1b_l; i++)
                fprintf(debugFile, "%.4f ", uc(i));
            fprintf(debugFile, "]\n");
            
            fprintf(debugFile,"t=%.3e, error[%d,%d]=[", t, nd1a_l, nd1b_l);
            for (int i=nd1a_l; i<=nd1b_l; i++ )
            {
                fprintf(debugFile, "%.4e ", abs(error(i)) );
            }
            fprintf(debugFile, "]\n");
            fclose(debugFile);
        }
    // get the global max error
    maxErr = getMaxValue( maxErr );
    // get max cpu time over all processors
    cpuTimeStep = getMaxValue( cpuTimeStep );

    // Gather results to processor 0 and print and output in matlab file
    Real *u_global = NULL;
    Real *x_global = NULL;
    Real *error_global = NULL;

    if (myRank ==0 )
    {
        // allocate global arrays
        const int nd1_global = Nx + 2*numghost +1;
        u_global = new Real[nd1_global];
        x_global = new Real[nd1_global];
        error_global = new Real[nd1_global];

        // gather the data from all processors
        // first copy data from rank 0
        for( int i=nd1a_l; i<=nd1b_l; i++ ) // i is the global index
        {   
            u_global[i + numghost] = uc(i);
            x_global[i + numghost] = x(i);
            error_global[i + numghost] = error(i);
        }

        // now receive from other processors
        for( int p=1; p<np; p++ )
        {
            int nx_l_p, n1a_l_p, n1b_l_p; // local values for processor p
            getLocalIndexBounds( p, np, Nx, nx_l_p, n1a_l_p, n1b_l_p ); // to determine the local bounds

            const int nd1a_p = n1a_l_p - numghost;
            const int nd1b_p = n1b_l_p + numghost;
            const int count = nd1b_p - nd1a_p + 1;

            Real *recv_u = new Real[count];
            Real *recv_x = new Real[count];
            Real *recv_error = new Real[count];

            MPI_Status status;
            MPI_Recv( recv_u, count, MPI_DOUBLE, p, 001, MPI_COMM_WORLD, &status );
            MPI_Recv( recv_x, count, MPI_DOUBLE, p, 002, MPI_COMM_WORLD, &status );
            MPI_Recv( recv_error, count, MPI_DOUBLE, p, 003, MPI_COMM_WORLD, &status );

            // copy into global arrays
            for( int i=nd1a_p; i<=nd1b_p; i++ ) // i is the global index
            {   
                int I = i + numghost;
                int local_i = i - nd1a_p; // I is the local index into recv arrays
                u_global[I] = recv_u[local_i];
                x_global[I] = recv_x[local_i];
                error_global[I] = recv_error[local_i];
            }

            delete [] recv_u;
            delete [] recv_x;
            delete [] recv_error;
        }
    }
    else 
    {
        // all other processors send their data to rank 0
        const int count = nd1b_l - nd1a_l + 1;
        MPI_Send( u_p[cur], count, MPI_DOUBLE, 0, 001, MPI_COMM_WORLD );
        MPI_Send( x_p, count, MPI_DOUBLE, 0, 002, MPI_COMM_WORLD );
        MPI_Send( error_p, count, MPI_DOUBLE, 0, 003, MPI_COMM_WORLD );
    }

    if (myRank == 0 )
    {  
                // writing the global arrays
        const int global_nd1a = -numghost;
        const int global_nd1b = Nx + numghost;
        printf("------------------- Solve the heat equation in 1D solution=%s --------------------\n", solutionName);
        printf("  np=%d, commOption=%d : %s\n",  np, commOption, commOption == 0 ? "blockSendReceive" : "nonBlockSendReceive");
        printf("  numGhost = %d, n1a=%d, n1b=%d, nd1a=%d, nd1b=%d\n", numghost, n1a, global_nd1b-numghost, global_nd1a, global_nd1b);
        printf("  numSteps = %d, Nx=%d, kappa=%g, tFinal=%g, boundaryCondtion(0,0)=%d, boundaryCondtion(1,0)=%d\n", 
            numSteps, Nx, kappa, tFinal, boundaryCondition(0,0), boundaryCondition(1,0));
        printf("np=%1d, numSteps=%4d, Nx=%3d, maxErr=%9.2e, cpu=%9.2e(s)\n",np,numSteps,Nx,maxErr,cpuTimeStep);
        printf("np=%d, myRank=%d, Gather solution: maxErr = %10.3e\n",np,myRank,maxErr);

        FILE *matlabFile = fopen(matlabFileName.c_str(),"w");
        fprintf(matlabFile,"%% File written by heat1dMPI.C\n");
        fprintf(matlabFile,"xa=%g; xb=%g; kappa=%g; t=%g; maxErr=%10.3e; cpuTimeStep=%10.3e;\n",xa,xb,kappa,tFinal,maxErr,cpuTimeStep);
        fprintf(matlabFile,"Nx=%d; dx=%14.6e; numGhost=%d; n1a=%d; n1b=%d; nd1a=%d; nd1b=%d;\n",Nx,dx,numghost,0, Nx, -numghost, Nx+numghost);
        fprintf(matlabFile,"solutionName=\'%s\';\n",solutionName);


        writeMatlabVector( matlabFile, x_global - global_nd1a, "x", global_nd1a, global_nd1b);
        writeMatlabVector( matlabFile, u_global - global_nd1a, "u", global_nd1a, global_nd1b);
        writeMatlabVector( matlabFile, error_global - global_nd1a, "err", global_nd1a, global_nd1b);
        
        fclose(matlabFile);
        printf("Wrote file %s\n\n",matlabFileName.c_str());

        // delete global arrays
        delete [] u_global;
        delete [] x_global;
        delete [] error_global;
    }

    // delete local arrays
    delete [] u_p[0];
    delete [] u_p[1];
    delete [] x_p;
    delete [] error_p;
    delete [] boundaryCondition_p;

    MPI_Finalize(); // for all processors
    return 0;
}






