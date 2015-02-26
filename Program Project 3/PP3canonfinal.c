#include<mpi.h>
#include<stdlib.h>
#include<math.h>
#include<stdio.h>
#include<string.h>
double **A, **B, **C;
double *a, *b, *c, *temple_a, *temple_b;
int dg, dl, dl2,p, sp;
int my_rank, my_row, my_col;
MPI_Status status;
void CannonAlgorithm(int n, double *a, double *b, double *c, MPI_Comm comm);
void MatrixMultiply(int n, double *a, double *b, double *c);
void creatMatA_B();
void scatter_A_B();
void collect_C();
void print(double **m,char *str);
int main(int argc, char *argv[])
{
   int i,j;
   double begin, last;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &p);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   MPI_Request reqs[4];
   sp = sqrt(p);


   if (sp*sp != p)
   {
      if (my_rank == 0)
	  printf("Number of processors is not a quadratic number!\n");
      MPI_Finalize();
      exit(1);
   }

   if (argc != 2)
   {
      if (my_rank == 0)
          printf("usage: mpirun -np ProcNum cannon MatrixDimension\n");
      MPI_Finalize();
      exit(1);
   }

   dg  = atoi(argv[1]); //the size of matrix(dim)
   dl  = dg / sp;       //the size of block matrix(dim)
   dl2 = dl * dl;       //the size of block matrix(dim*dim)

   //calculate the coordinate according to the rank of process
   //for example, sp=4,when rank=6(0<rank<16), my_col=2, my_row=1; thus the coordinate is (1,2);
   //****
   //*?**
   //****
   //****
   my_col =  my_rank % sp ;
   my_row = (my_rank-my_col) / sp ;


   a = (double *)malloc( dl2 * sizeof(double) );
   b = (double *)malloc( dl2 * sizeof(double) );
   c = (double *)malloc( dl2 * sizeof(double) );


   for(i=0; i<dl2 ; i++)
     c[i] = 0.0;


   temple_a = (double *)malloc( dl2 * sizeof(double) );
   temple_b = (double *)malloc( dl2 * sizeof(double) );

  if (my_rank == 0)
       begin = MPI_Wtime();

  if (my_rank == 0)
   {

      A = (double **)malloc( dg * sizeof(double*) );
      B = (double **)malloc( dg * sizeof(double*) );
      C = (double **)malloc( dg * sizeof(double*) );

      for(i=0; i<dg; i++)
      {
         A[i] = (double *)malloc( dg * sizeof(double) );
         B[i] = (double *)malloc( dg * sizeof(double) );
         C[i] = (double *)malloc( dg * sizeof(double) );
      }
      creatMatA_B();
      scatter_A_B();
   }
   else
   {

       MPI_Irecv(a, dl2, MPI_DOUBLE, 0 , 1, MPI_COMM_WORLD, &reqs[0]);
       MPI_Irecv(b, dl2, MPI_DOUBLE, 0 , 2, MPI_COMM_WORLD, &reqs[1]);
       for(j=0;j<2;j++)
      {
        MPI_Wait(&reqs[j],&status);
      }
   }

 CannonAlgorithm(dg,a, b,c, MPI_COMM_WORLD);

   if(my_rank == 0)
   {
     collect_C();
     if (dg<20)
   {
     print(A,"random matrix A : \n");
	 print(B,"random matrix B : \n");
	 print(C,"Matrix C = A * B : \n");
   }

  } else
   {
      MPI_Isend(c,dl2,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&reqs[2]);
      MPI_Wait(&reqs[2],&status);

   }

   MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank==0)
      {
       last = MPI_Wtime();
       printf("Elapsed Time of Computation is: %f\n", last - begin);
      }

   MPI_Finalize();

   return 0;
}

void creatMatA_B()
{
   int i,j;
    srand((unsigned int)time(NULL));
    for(i=0; i<dg; i++)
      for(j=0; j<dg ; j++)
	  {
    	    A[i][j] = rand();
            B[i][j] = rand();
            C[i][j] = 0.0;
	  }
}
void scatter_A_B()
{
   int i,j,k,l;
   int p_imin,p_imax,p_jmin,p_jmax;
   // dg  = atoi(argv[1]); //the size of matrix(dim)
   // dl  = dg / sp;       //the size of block matrix(dim)
   // dl2 = dl * dl;       //the size of block matrix(dim*dim)
   for(k=0; k<p; k++)

   {

	  p_jmin = (k % sp) * dl;
  	  p_jmax = (k % sp + 1) * dl-1;
	  p_imin = (k - (k % sp))/sp * dl;
	  p_imax = ((k - (k % sp))/sp +1) *dl -1;
               l = 0;
	 //range of submatrix(imin,jmin):(imax,jmax).
      //split the matrix AandB into buffer, devide the whole matirx into submatrix
      for(i=p_imin; i<=p_imax; i++)
      {
      	  for(j=p_jmin; j<=p_jmax; j++)
      	  {
          temple_a[l] = A[i][j];
	      temple_b[l] = B[i][j];
	      l++;
          }
      }

      //scatter the submatrix to p processors.
      if(k==0)
      {
         memcpy(a, temple_a, dl2 * sizeof(double));
	     memcpy(b, temple_b, dl2 * sizeof(double));
      } else
      {
      MPI_Send(temple_a, dl2, MPI_DOUBLE, k, 1, MPI_COMM_WORLD);
	  MPI_Send(temple_b, dl2, MPI_DOUBLE, k, 2, MPI_COMM_WORLD);
      }
   }
}
void collect_C()
{
   int i,j,i2,j2,k;
   int p_imin,p_imax,p_jmin,p_jmax;


   for (i=0;i<dl;i++)
	 for(j=0;j<dl;j++)
	   C[i][j]=c[i*dl+j];

   for (k=1;k<p;k++)
   {

       MPI_Recv(c, dl2, MPI_DOUBLE, k, 1, MPI_COMM_WORLD, &status);

       p_jmin = (k % sp) *dl;
       p_jmax = (k % sp + 1) *dl-1;
       p_imin =  (k - (k % sp))/sp     *dl;
       p_imax = ((k - (k % sp))/sp +1) *dl -1;

       i2=0;

       for(i=p_imin; i<=p_imax; i++)
       {
           j2=0;
           for(j=p_jmin; j<=p_jmax; j++)
           {
               C[i][j]=c[i2*dl+j2];
               j2++;
           }
           i2++;
       }
   }
}
void print(double **m,char *str)
{
   int i,j;
   printf("%s",str);
   if (dg< 20)
   {
    for(i=0;i<dg;i++)
    {
       for(j=0;j<dg;j++)
           printf("%15.1f\t",m[i][j]);
           printf("\n");
    }
   }
   printf("\n");
}
void CannonAlgorithm(int n, double *a, double *b, double *c, MPI_Comm comm)
{
      int i;
      int nlocal;
      int npes, dims[2], periods[2];
      int myrank, my2drank, mycoords[2];
      int uprank, downrank, leftrank, rightrank, coords[2];
      int shiftsource, shiftdest;
      MPI_Status status;
      MPI_Comm comm_2d;

      // Print Matrix
//      if( gRank == 0 )
//      {
//          printMatrix(a,n,n);
//          printf("\n");
//          printMatrix(b,n,n);
//      }

      /* Get the communicator related information */
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &myrank);

      /* Set up the Cartesian topology */
      dims[0] = dims[1] = sqrt(npes);

      /* Set the periods for wraparound connections */
      periods[0] = periods[1] = 1;

      /* Create the Cartesian topology, with rank reordering */
      MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d);

      /* Get the rank and coordinates with respect to the new topology */
      MPI_Comm_rank(comm_2d, &my2drank);
      MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);

      // There needs to be some switching done here
      /* Compute ranks of the up and left shifts */
      MPI_Cart_shift(comm_2d, 1, -1, &rightrank, &leftrank);
      MPI_Cart_shift(comm_2d, 0, -1, &downrank, &uprank);

      /* Determine the dimension of the local matrix block */
      nlocal = n/dims[0];

      /* Perform the initial matrix alignment. First for A and then for B */
//      MPI_Cart_shift(comm_2d, 0, -mycoords[0], &shiftsource, &shiftdest);
      MPI_Cart_shift(comm_2d, 1, -mycoords[0], &shiftsource, &shiftdest);

      MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE, shiftdest,
          1, shiftsource, 1, comm_2d, &status);

//      MPI_Cart_shift(comm_2d, 1, -mycoords[1], &shiftsource, &shiftdest);
      MPI_Cart_shift(comm_2d, 0, -mycoords[1], &shiftsource, &shiftdest);
      MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE,
          shiftdest, 1, shiftsource, 1, comm_2d, &status);

      /* Get into the main computation loop */
    for (i=0; i<dims[0]; i++)
    {
//        if( gRank == 0 )
//        {
//            printMatrix(b,n,n);
//            puts("");
//        }
//        //Print C before
//        puts("Before: C Matrix");
//        printMatrix(c,n,n);
        MatrixMultiply(nlocal, a, b, c); /*c=c+a*b*/


//        //Print C after
//        puts("After: C Matrix");
//        printMatrix(c,n,n);

        /* Shift matrix a left by one */
        MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE,
        leftrank, 1, rightrank, 1, comm_2d, &status);
//        printf("Proc1: %d\nProc2: %d Dims:%d\n",my2drank,gRank,i);

        /* Shift matrix b up by one */
        MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE,
        uprank, 1, downrank, 1, comm_2d, &status);

//        if( gRank == 0 )
//        {
//            printMatrix(b,n,n);
//            puts("");
//        }
    }

      /* Restore the original distribution of a and b */
//      MPI_Cart_shift(comm_2d, 0, +mycoords[0], &shiftsource, &shiftdest);
      MPI_Cart_shift(comm_2d, 1, +mycoords[0], &shiftsource, &shiftdest);
      MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE,
          shiftdest, 1, shiftsource, 1, comm_2d, &status);

//      MPI_Cart_shift(comm_2d, 1, +mycoords[1], &shiftsource, &shiftdest);
      MPI_Cart_shift(comm_2d, 0, +mycoords[1], &shiftsource, &shiftdest);
      MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE,
          shiftdest, 1, shiftsource, 1, comm_2d, &status);

      MPI_Comm_free(&comm_2d); /* Free up communicator */
}

    /* This function performs a serial matrix-matrix multiplication c = a*b */
void MatrixMultiply(int n, double *a, double *b, double *c)
{
    int i, j, k;

    for (i=0; i<n; i++)
        for (j=0; j<n; j++)
            for (k=0; k<n; k++)
                c[i*n+j] += a[i*n+k]*b[k*n+j];
}
