// =====================================================================
//
// HEAT EQUATION IN TWO DIMENSIONS
// Solve with A++ arrays
//
// =====================================================================
#include "A++.h"
#include <mpi.h>
#include "getLocalIndexBounds.h" // required to determine the distribution of the grid points

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <unistd.h> 

// define some types
typedef double Real;
typedef doubleSerialArray RealArray;
typedef intSerialArray IntegerArray;
#include <string>
using std::string;      // to avoid writing std::string everywhere
using std::max;         // to avoid ambiguity with std::max and fmax and just use max

#include <float.h>
#include <limits.h>
#define REAL_EPSILON DBL_EPSILON
#define REAL_MIN DBL_MIN

// getCPU() : Return the current wall-clock time in seconds
// #include "getCPU.h"

// include commands tp parse command line arguments
#include "parseCommand.h"

// function to write an array to a matlab reabable file:
#include "writeMatlabArray.h"

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
void exchangeGhostPoints(RealArray &u, int n1a, int n1b, int n2a_l, int n2b_l,
                         int nd1a, int nd1b, int nd2a_l, int nd2b_l,
                          int myRank, int np, int commOption, MPI_Comm comm = MPI_COMM_WORLD)
{   
    // For the 2d arrays: we need to send/recv entire rows (all x values at a give ny ghostpoints)
    int rowSize = nd1b - nd1a + 1; // number of elements in a row 
    Real *sendBufferDown = new Real[rowSize];
    Real *sendBufferUp = new Real[rowSize];
    Real *recvBufferDown = new Real[rowSize];
    Real *recvBufferUp = new Real[rowSize];

    if (commOption == 0) // Blocked Sending/Receiving
    {  
        MPI_Status status;

        // Packing the send buffers
        if (myRank > 0) // has down neighbor
        {
            for ( int i1 = nd1a; i1 <= nd1b; i1++ )
            {
                sendBufferDown[i1 - nd1a] = u(i1, n2a_l); // copy the downmost real row
            }
        }
        if (myRank < np - 1) // has up neighbor
        {
            for ( int i1 = nd1a; i1 <= nd1b; i1++ )
            {
                sendBufferUp[i1 - nd1a] = u(i1, n2b_l); // copy the upmost real row
            }
        }

        // Exchange with down neighbour
        if (myRank > 0) 
        {
            MPI_Sendrecv(sendBufferDown, rowSize, MPI_DOUBLE, myRank - 1, 1,      // send down
                        recvBufferDown, rowSize, MPI_DOUBLE, myRank - 1, 0,   // receive from down
                        comm, &status);
            // Unpacking the recv buffer
            for ( int i1 = nd1a; i1 <= nd1b; i1++ )
            {
                u(i1, n2a_l - 1) = recvBufferDown[i1 - nd1a]; // copy into down ghost row
            }
        }
        // Exchange with up neighbor
        if (myRank < np - 1) 
        {
            MPI_Sendrecv(sendBufferUp, rowSize, MPI_DOUBLE, myRank + 1, 0,      // send up
                        recvBufferUp, rowSize, MPI_DOUBLE, myRank + 1, 1,   // receive from up
                        comm, &status);
            // Unpacking the recv buffer
            for ( int i1 = nd1a; i1 <= nd1b; i1++ )
            {
                u(i1, n2b_l + 1) = recvBufferUp[i1 - nd1a]; // copy into up ghost row
            }
        }
    }
    else 
    {
        MPI_Request requests[4];
        int numRequests =0;

        // The non-blocking receives first
        if (myRank > 0) // has down neighbor
        {
            MPI_Irecv( recvBufferDown, rowSize, MPI_DOUBLE, myRank -1, 0, comm, &requests[numRequests++]); // receive down ghost point
        }
        if (myRank < np -1) // has up neighbor
        {
            MPI_Irecv( recvBufferUp, rowSize, MPI_DOUBLE, myRank +1, 1, comm, &requests[numRequests++]); // receive up ghost point   
        }
        // Now the non-blocking packs and sends
        if (myRank < np -1) // has up neighbor
        {
            for ( int i1 = nd1a; i1 <= nd1b; i1++ )
            {
                sendBufferUp[i1 - nd1a] = u(i1, n2b_l); // copy the upmost real row
            }
            MPI_Isend( sendBufferUp, rowSize, MPI_DOUBLE, myRank +1, 0, comm, &requests[numRequests++]); // send upmost real point
        }
        if (myRank > 0) // has down neighbor
        {
            for ( int i1 = nd1a; i1 <= nd1b; i1++ )
            {
                sendBufferDown[i1 - nd1a] = u(i1, n2a_l); // copy the downmost real row
            }
            MPI_Isend( sendBufferDown, rowSize, MPI_DOUBLE, myRank -1, 1, comm, &requests[numRequests++]); // send downmost real point
        }
        // Wait for all non-blocking operations to complete
        MPI_Waitall( numRequests, requests, MPI_STATUSES_IGNORE );

        // Unpacking the recv buffers
        if (myRank > 0) // has down neighbor
        {
            for ( int i1 = nd1a; i1 <= nd1b; i1++ )
            {
                u(i1, n2a_l - 1) = recvBufferDown[i1 - nd1a]; // copy into down ghost row
            }
        }
        if (myRank < np -1) // has up neighbor
        {
            for ( int i1 = nd1a; i1 <= nd1b; i1++ )
            {
                u(i1, n2b_l + 1) = recvBufferUp[i1 - nd1a]; // copy into up ghost row
            }
        }
    }
    delete [] sendBufferDown;
    delete [] sendBufferUp;
    delete [] recvBufferDown;
    delete [] recvBufferUp;
}
// --- Declare the fortran routine as a "C" function
// Some compilers add an under-score to the name of "C" and Fortran routines
// Note also that the fortran name is all lowercase
#define heat2dUpdate heat2dupdate_

extern "C"
{
    void heat2dUpdate( const int & n1a, const int & n1b, const int & n2a, const int & n2b,
                      const int & nd1a, const int & nd1b, const int & nd2a, const int & nd2b,
                      Real & un, const Real &u, const Real & rx, const Real &ry );
}


int
main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv); // Initialize MPI
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank); // Get the processor number
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np); // Get the number of processors
    int version, subversion;
    MPI_Get_version(&version, &subversion);

    // Get processor/node name
    char procName[MPI_MAX_PROCESSOR_NAME];
    int resultlen;
    MPI_Get_processor_name(procName, &resultlen);

    pid_t pid = getpid(); // get the process ID

    // if (myRank==0)
    // {
    //     printf("Usage: heat2d -nx=i -ny=i -debug=i -tFinal=f -commOption=i -matlabFineName=s]\n" //saves matlabfile if called
    //            "    commOption : 0=blockSendReceive, 1=nonBlockSendReceive\n"
    //            "    matlabFileName.m : save results to this file.\n");
    // }
    // -------------------------------- Define the exact solution -----------------------------
    #define trueSolution 1 // To solve the head equation 
    #define manufacturedSolution 2 // To use the manufactured solution with source term

    #ifndef SOLUTION
        // #define SOLUTION trueSolution
        #define SOLUTION manufacturedSolution
    #endif
    // ----------------------------------------------------------------------------------------
    const char *sol;
    #if SOLUTION == trueSolution
        sol = "TS";
    #else
        sol = "Polynomial MS";
    #endif

    const Real pi = 4.*atan2(1.,1.);

    ios::sync_with_stdio(); // Synchronize C++ and C I/O subsystems
    Index::setBoundsCheck(on); // Turn on A++ array bounds checking
    // if (myRank==0)
    // {
    //     printf("Usage: heat2d -nx=<i> -option=[0|1|2|3] -tFinal=<f> -debug=<i> -saveMatlab=[0|1|2] matlabFile=<s>\n"
    //        "   option : 0=scalarIndexing, 1=arrayIndexing, 2=cIndexing, 3=fortranRoutine\n");
    // }
    enum BoundaryConditionsEnum
    {
        periodic=-1,
        dirichlet=1,
        neumann=2
    };

    enum OptionsEnum
    {
        scalarIndexing=0,
        arrayIndexing =1,
        cIndexing =2,
        fortranRoutine=3
    };

    int option=cIndexing;
    int commOption =1; 
    const int numberOfDimensions=2;

    int debug = 0;
    Real kappa=.1;
    Real xa=0., xb=1.; // domain is [xa,xb] X [ya,yb]
    Real ya=0., yb=1.;

    Real tFinal=.5;
    int nx=100, ny=nx;

    int saveMatlab=0; // 1 = save a matlab file, 2=save solution too
    string matlabFileName = "heat2d.m";
    
    string line;
    for( int i=1; i<argc; i++ )
    {
        line=argv[i];
        // printf("Input: argv[%d] = [%s]\n",i,line.c_str());
        if( parseCommand( line,"-nx=",nx) )
        {
            ny=nx;
        }
        else if( parseCommand( line,"-debug=",debug) ){}
        else if( parseCommand( line,"-option=",option) ){}
        else if( parseCommand( line, "-tFinal=",tFinal) ){}
        else if( parseCommand( line,"-commOption=",commOption) ){}
        else if( parseCommand( line,"-saveMatlab=",saveMatlab) ){}
        else if( parseCommand( line,"-matlabFileName=",matlabFileName) ){}

    }

    const int numGhost=1;
    const int n1a = 0;
    const int n1b = n1a + nx;
    const int nd1a = n1a-numGhost;
    const int nd1b = n1b+numGhost;
    const int nd1 = nd1b-nd1a+1;

    // Global y-direction indices
    const int n2a = 0;
    const int n2b = n2a + ny;
    const int nd2a = n2a-numGhost;
    const int nd2b = n2b+numGhost;

    // Distributing the y-direction grid across the processors
    int ny_l, n2a_l, n2b_l; // local values for this processor
    getLocalIndexBounds( myRank, np, ny, ny_l, n2a_l, n2b_l ); // to determine the local bounds
    const int nd2a_l = n2a_l - numGhost;
    const int nd2b_l = n2b_l + numGhost;
    const int nd2_l = nd2b_l - nd2a_l + 1;

    IntegerArray gridIndexRange(2,numberOfDimensions);
    IntegerArray dimension(2,numberOfDimensions);
    IntegerArray boundaryCondition(2,numberOfDimensions);

    gridIndexRange(0,0)=n1a; gridIndexRange(1,0)=n1b;
    gridIndexRange(0,1)=n2a; gridIndexRange(1,1)=n2b;

    dimension(0,0)=nd1a; dimension(1,0)=nd1b;
    dimension(0,1)=nd2a; dimension(1,1)=nd2b;

    boundaryCondition(0,0)=dirichlet; // left
    boundaryCondition(1,0)=dirichlet; // right

    boundaryCondition(0,1)=dirichlet; // bottom
    boundaryCondition(1,1)=dirichlet; // top

    string optionName = option==scalarIndexing ? "scalarIndexing" :
                        option==arrayIndexing ? "arrayIndexing " :
                        option==cIndexing ? "cIndexing     " :
                        option==fortranRoutine ? "fortranRoutine" : "unknown";
    
    if (myRank==0)
    {
        printf("----- Solve the Heat Equation in two dimensions ------\n");
        printf("      option=%d : %s \n",option, optionName.c_str());
        printf("      saveMatlab=%d, matlabFileName=%s \n",saveMatlab,matlabFileName.c_str());
        printf("      kappa=%.3g, nx=%d, ny=%d, tFinal=%6.2f\n",kappa,nx,ny,tFinal);
        printf("      Solution: %s\n", sol);
    }

    // printf("Rank %d: Local y-range [%d,%d], with ghosts [%d,%d]\n", 
    //        myRank, n2a_l, n2b_l, nd2a_l, nd2b_l);

    // Grid points
    Range Rx(nd1a,nd1b), Ry_l(nd2a_l,nd2b_l);
    RealArray x(Rx,Ry_l,2);

    Real dx[2];
    dx[0] = (xb-xa)/nx;
    dx[1] = (yb-ya)/ny;

    int i1,i2;
    for( i2=nd2a_l; i2<=nd2b_l; i2++ )
        for( i1=nd1a; i1<=nd1b; i1++ )
        {
            x(i1,i2,0) = xa + (i1-n1a)*dx[0];
            x(i1,i2,1) = ya + (i2-n2a)*dx[1];
        }

    # if SOLUTION==trueSolution
        const Real kx=2., ky=3;
        const Real kxp=kx*pi;
        const Real kyp=ky*pi;
        #define UTRUE(x,y,t) sin( kxp*(x) )*sin( kyp*(y) )*exp(-kappa*(kxp*kxp+kyp*kyp)*(t) )

    # elif SOLUTION==manufacturedSolution // polynomial manufactured solution
        static const Real c0=.2, c1=.1, c2=.3;
        static const Real b0=1., b1=.5, b2=.25;
        static const Real a0=1., a1=.3, a2=0.;
        #define UTRUE(x,y,t) (b0 + (x)*( b1 + (x)*b2 ))*(c0 + (y)*( c1 + (y)*c2 ))*( a0 + (t)*( a1 + (t)*a2 ) )
        #define UTRUET(x,y,t) (b0 + (x)*( b1 +(x)*b2 ))*(c0 + (y)*( c1 + (y)*c2 ))*( a1 + 2.*(t)*a2 )
        #define UTRUEXX(x,y,t) ( 2.*b2 )*(c0 + (y)*( c1 + (y)*c2 ))*( a0 + (t)*( a1 + (t)*a2 ) )
        #define UTRUEYY(x,y,t) (b0 + (x)*( b1 + (x)*b2 ))*( 2.*c2 )*( a0 + (t)*( a1 + (t)*a2 ) )
        #define FORCE(x,y,t) ( UTRUET(x,y,t) - kappa*( UTRUEXX(x,y,t)+UTRUEYY(x,y,t) ) )
    #else
        printf("ERROR: unknown SOLUTION=%d\n",SOLUTION);
        abort();
    #endif

    // we store two time levels
    RealArray ua[2];
    ua[0].redim(Rx,Ry_l); ua[0]=0.;
    ua[1].redim(Rx,Ry_l); ua[1]=0.;

    // initial conditions
    RealArray &u0 = ua[0];
    Real t=0.;
    for( i2=nd2a_l; i2<=nd2b_l; i2++ )
        for( i1=nd1a; i1<=nd1b; i1++ )
        {
            u0(i1,i2)= UTRUE(x(i1,i2,0),x(i1,i2,1),t);
        }
    
    // starting a debug file for each processor 
    FILE *debugFile = nullptr;
    if (debug > 0)
    {
        char debugFileName[256];
        sprintf (debugFileName, "debug/heat2d_debugNp%d_proc_%d.txt",np, myRank);
        debugFile = fopen (debugFileName, "w");
        fprintf (debugFile, " ----- Solve the Heat Equation in two dimensions, np=%d, myRank=%d ------\n", np, myRank);
        fprintf (debugFile,"   procName=[%s], pid=%d, MPI version %d.%d\n",procName, pid, version, subversion);
        fprintf (debugFile, "   kappa=%.3g, nx=%d, ny=%d, tFinal=%6.2f, %s\n",kappa,nx,ny,tFinal, sol);
        fprintf (debugFile, "   Local bounds: ny= %d, ny_l= %d, [n2a_1,n2b_l]=[%d,%d]\n", ny, ny_l, n2a_l, n2b_l);
        fclose (debugFile);
    }

    // Time-step restriction:
    // Forward Euler: kappa*dt*( 1/dx^2 + 1/dy^2 ) < cfl*.5
    Real cfl=.9;
    Real dt = cfl*(.5/kappa)/( 1./(dx[0]*dx[0]) + 1./(dx[1]*dx[1]) );

    int numSteps=ceil(tFinal/dt);
    dt = tFinal/numSteps; // adjust dt to reach the final time

    Real rx = kappa*(dt/(dx[0]*dx[0]));
    Real ry = kappa*(dt/(dx[1]*dx[1]));
    int cur=0;
    int i,n;
    Index I1=Range(n1a,n1b);
    Index I2=Range(n2a_l,n2b_l);

    Real cpu1 = getCPU();
    for( n=0; n<numSteps; n++ )
    {
        t = n*dt; // cur time

        int next = (cur+1) % 2;
        RealArray & u = ua[cur];
        RealArray & un = ua[next];

        if( option==scalarIndexing )
        {
            for( i2=n2a_l; i2<=n2b_l; i2++ )
                for( i1=n1a; i1<=n1b; i1++ )
                {
                    un(i1,i2) = u(i1,i2) + rx*( u(i1+1,i2) -2.*u(i1,i2) + u(i1-1,i2) )
                                         + ry*( u(i1,i2+1) -2.*u(i1,i2) + u(i1,i2-1) );
                }
        }
        else if( option==arrayIndexing )
        {
            un(I1,I2) = u(I1,I2) + rx*( u(I1+1,I2) -2.*u(I1,I2) + u(I1-1,I2) )
                                 + ry*( u(I1,I2+1) -2.*u(I1,I2) + u(I1,I2-1) );
        }
        else if( option==cIndexing )
        {
            // Index as C arrays
            const double *u_p = u.getDataPointer();
            double *un_p = un.getDataPointer();
            #define U(i1,i2) u_p[(i1-nd1a)+nd1*(i2-nd2a_l)]
            #define UN(i1,i2) un_p[(i1-nd1a)+nd1*(i2-nd2a_l)]

            for( i2=n2a_l; i2<=n2b_l; i2++ )
                for( i1=n1a; i1<=n1b; i1++ )
                {
                    UN(i1,i2) = U(i1,i2) + rx*( U(i1+1,i2) -2.*U(i1,i2) + U(i1-1,i2) )
                                         + ry*( U(i1,i2+1) -2.*U(i1,i2) + U(i1,i2-1) )
                                         + dt*FORCE( x(i1,i2,0), x(i1,i2,1), t );
                }
            
            #undef U
            #undef UN

        }
        else if( option==fortranRoutine )
        {
            // call a fortran routine
            // Note: pass first element of un and u arrays to Fortran (call by reference)
            if( true )
            {
                const double *u_p = u.getDataPointer();
                double *un_p = un.getDataPointer();
                heat2dUpdate( n1a,n1b,n2a,n2b,
                             u.getBase(0),u.getBound(0), u.getBase(1),u.getBound(1), // pass array dimensions
                             *un_p, *u_p, rx,ry);
            }
            else
            {
                // this will also work -- pass first element of un and u
                heat2dUpdate( n1a,n1b,n2a,n2b,
                             u.getBase(0),u.getBound(0), u.getBase(1),u.getBound(1), // pass array dimensions
                             un(nd1a,nd2a), u(nd1a,nd2a), rx,ry);
            }
        }
        else
        {
            printf("ERROR: unknown option=%d\n",option);
            abort();
        }


        // --- boundary conditions ---
        for(int axis = 0; axis < numberOfDimensions; axis++)
        for(int side = 0; side <= 1; side++)
        {
            int is = 1 - 2*side;
            
            if(boundaryCondition(side, axis) == dirichlet)
            {
                if(axis == 0)
                { 
                    // X-direction - ALL PROCESSORS
                    i1 = gridIndexRange(side, axis);
                    int i1g = i1 - is;
                    
                    for(i2 = nd2a_l; i2 <= nd2b_l; i2++)  // ← Uses LOCAL bounds
                    {
                        un(i1, i2) = UTRUE(x(i1, i2, 0), x(i1, i2, 1), t+dt);
                        un(i1g, i2) = 3.*un(i1, i2) - 3.*un(i1+is, i2) + un(i1+2*is, i2);
                    }
                }
                else
                { 
                    // Y-direction - on the Global Boundarys
                    bool ownsBoundary = (side == 0 && myRank == 0) ||
                                    (side == 1 && myRank == np-1);
                    
                    if(ownsBoundary)  // ← MPI guard
                    {
                        i2 = gridIndexRange(side, axis);
                        int i2g = i2 - is;
                        
                        for(i1 = nd1a; i1 <= nd1b; i1++)
                        {
                            un(i1, i2) = UTRUE(x(i1, i2, 0), x(i1, i2, 1), t+dt);
                            un(i1, i2g) = 3.*un(i1, i2) - 3.*un(i1, i2+is) + un(i1, i2+2*is);
                        }
                    }
                }
            }
        }
        // --- exchange ghost points with neighboring processors ---
        exchangeGhostPoints(un, n1a, n1b, n2a_l, n2b_l, nd1a, nd1b, nd2a_l, nd2b_l,
                           myRank, np, commOption, MPI_COMM_WORLD);

        if (debug > 1)
        {
            // Compute local max error at this time step
            Real stepMaxErr = 0.;
            for(i2 = n2a_l; i2 <= n2b_l; i2++)
                for(i1 = n1a; i1 <= n1b; i1++)
                {
                    Real error = fabs(un(i1,i2) - UTRUE(x(i1,i2,0), x(i1,i2,1), t+dt));
                    stepMaxErr = max(error, stepMaxErr);
                }
            stepMaxErr = getMaxValue(stepMaxErr, -1, MPI_COMM_WORLD);
            
            if (myRank == 0)
                printf("  step %4d: t=%8.3e maxErr=%10.3e\n", n+1, t+dt, stepMaxErr);
        }

        cur = next;
    }
    Real cpuTimeStep = getCPU()-cpu1;

    // --- compute errors ---
    RealArray & uc = ua[cur];
    RealArray err(Rx,Ry_l);

    Real maxErr=0., maxNorm=0.;
    for( i2=n2a_l; i2<=n2b_l; i2++ )
        for( i1=n1a; i1<=n1b; i1++ )
        {
            err(i1,i2) = fabs(uc(i1,i2) - UTRUE(x(i1,i2,0),x(i1,i2,1),tFinal));
            maxErr = max(err(i1,i2),maxErr);
            maxNorm = max(uc(i1,i2),maxNorm);
        }
    // get the max error over all processors
    maxErr = getMaxValue(maxErr,0,MPI_COMM_WORLD);
    maxNorm = getMaxValue(maxNorm,0,MPI_COMM_WORLD);
    maxErr /= max(maxNorm,REAL_MIN); // relative error

    if (myRank == 0)
    {
        printf("np=%2d: numSteps=%d nx=%d maxNorm=%8.2e maxRelErr=%8.2e cpuTimeStep=%9.2e(s)\n",
                np, numSteps, nx, maxNorm, maxErr, cpuTimeStep);
    }

    if( nx<=10 )
    {
        uc.display("ua[cur]");
        err.display("err");
    }

    // --- OPTIONALLY write a matlab file for plotting in matlab ---
    if( saveMatlab )
    {
        // Pack local data (interior points only, no ghosts)
        int localSize = (n1b - n1a + 1) * (n2b_l - n2a_l + 1);
        Real *localU = new Real[localSize];
        Real *localErr = new Real[localSize];
        Real *localX = new Real[localSize];
        Real *localY = new Real[localSize];
        
        int idx = 0;
        for(i2 = n2a_l; i2 <= n2b_l; i2++)
            for(i1 = n1a; i1 <= n1b; i1++)
            {
                localU[idx] = uc(i1, i2);
                localErr[idx] = err(i1, i2);
                localX[idx] = x(i1, i2, 0);
                localY[idx] = x(i1, i2, 1);
                idx++;
            }

        // Gather sizes from all processors
        int *recvCounts = nullptr;
        int *displs = nullptr;
        Real *globalU = nullptr;
        Real *globalErr = nullptr;
        Real *globalX = nullptr;
        Real *globalY = nullptr;

        if (myRank == 0)
        {
            recvCounts = new int[np];
            displs = new int[np];
        }

        MPI_Gather(&localSize, 1, MPI_INT, recvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (myRank == 0)
        {
            displs[0] = 0;
            for(int p = 1; p < np; p++)
                displs[p] = displs[p-1] + recvCounts[p-1];
            
            int totalSize = displs[np-1] + recvCounts[np-1];
            globalU = new Real[totalSize];
            globalErr = new Real[totalSize];
            globalX = new Real[totalSize];
            globalY = new Real[totalSize];
        }

        // Gather all data to rank 0
        MPI_Gatherv(localU, localSize, MPI_DOUBLE, globalU, recvCounts, displs, 
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localErr, localSize, MPI_DOUBLE, globalErr, recvCounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localX, localSize, MPI_DOUBLE, globalX, recvCounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localY, localSize, MPI_DOUBLE, globalY, recvCounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (myRank == 0)
        {
            // Compute error on gathered data
            Real maxErrGathered = 0., maxNormGathered = 0.;
            int idx = 0;
            for(int i2 = n2a; i2 <= n2b; i2++)
                for(int i1 = n1a; i1 <= n1b; i1++)
                {
                    Real trueVal = UTRUE(globalX[idx], globalY[idx], tFinal);
                    Real error = fabs(globalU[idx] - trueVal);
                    maxErrGathered = max(error, maxErrGathered);
                    maxNormGathered = max(fabs(globalU[idx]), maxNormGathered);
                    idx++;
                }
            maxErrGathered /= max(maxNormGathered, REAL_MIN);
            
            printf("np=%2d: After GATHER: maxNorm=%8.2e maxRelErr=%8.2e\n",
                np, maxNormGathered, maxErrGathered);
        }

        // Write MATLAB file on rank 0
        
        if (myRank == 0)
        {   
            FILE *matlabFile = nullptr;
            char fullPath[256];
            sprintf(fullPath, "matlab/heat2d_np%d.m", np);  
            matlabFile = fopen(fullPath, "w");
            if (matlabFile)
            {   
                fprintf(matlabFile, "%% File written by heat2d (MPI version)\n");
                fprintf(matlabFile, "xa=%g; xb=%g; ya=%g; yb=%g; kappa=%g; tFinal=%g; maxErr=%10.3e; cpuTimeStep=%10.3e;\n", xa, xb, ya, yb, kappa, tFinal, maxErr, cpuTimeStep);
                fprintf(matlabFile,"n1a=%d; n1b=%d; nd1a=%d; nd1b=%d;\n",n1a,n1b,nd1a,nd1b);
                fprintf(matlabFile,"n2a=%d; n2b=%d; nd2a=%d; nd2b=%d;\n",n2a,n2b,nd2a,nd2b);
                fprintf(matlabFile,"dx=%14.6e; dy=%14.6e; numGhost=%d;\n",dx[0],dx[1],numGhost);
                fprintf(matlabFile,"option=%d; optionName='%s';\n",option,optionName.c_str());
                if (saveMatlab > 1)
                {
                    int totalSize = displs[np-1] + recvCounts[np-1];
                    // resizing and writing global arrays for matlab
                    fprintf(matlabFile, "\n%% Global solution data\n");
                    
                    fprintf(matlabFile, "xGlobal = [\n");
                    for(int i = 0; i < totalSize; i++)
                        fprintf(matlabFile, "%.15e\n", globalX[i]);
                    fprintf(matlabFile, "];\n");
                    
                    fprintf(matlabFile, "yGlobal = [\n");
                    for(int i = 0; i < totalSize; i++)
                        fprintf(matlabFile, "%.15e\n", globalY[i]);
                    fprintf(matlabFile, "];\n");
                    
                    fprintf(matlabFile, "uGlobal = [\n");
                    for(int i = 0; i < totalSize; i++)
                        fprintf(matlabFile, "%.15e\n", globalU[i]);
                    fprintf(matlabFile, "];\n");
                    
                    fprintf(matlabFile, "errGlobal = [\n");
                    for(int i = 0; i < totalSize; i++)
                        fprintf(matlabFile, "%.15e\n", globalErr[i]);
                    fprintf(matlabFile, "];\n");
                }
                
                fclose(matlabFile);
                printf("Wrote file [matlab/%s]\n", matlabFileName.c_str());
            }
            else
            {
                printf("ERROR: Could not open file [%s] for writing\n", matlabFileName.c_str());
            }
            
            delete[] recvCounts;
            delete[] displs;
            delete[] globalU;
            delete[] globalErr;
            delete[] globalX;
            delete[] globalY;
        }
        
        delete[] localU;
        delete[] localErr;
        delete[] localX;
        delete[] localY;
    }

    MPI_Finalize();
    return 0;
}