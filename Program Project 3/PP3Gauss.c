#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "math.h"
#define a(x,y) a[x*M+y]
#define b(x) b[x]
#define A(x,y) A[x*M+y]
#define B(x) B[x]
#define floatsize sizeof(float)
#define intsize sizeof(int)
float *x;
int M;
int N;
int m;
float *A;
float *B;
double starttime;
double time1;
double time2;
int my_rank;
int p;
int l;
MPI_Status status;
MPI_Request request1;

void initialize_inputs() {
    int row, col;

   // printf("\nInitializing...\n");
    for (col = 0; col < M; col++) {
        for (row = 0; row < M; row++) {
            A[row*M+col] = (float)rand()/ 32768.0;
        }
        B[col] = (float)rand()/ 32768.0;
        x[col] = 0.0;
    }

}

/* Print input matrices */
void print_inputs() {
    int row, col;

    if (M < 10) {
        printf("\nA =\n\t");
        for (row = 0; row < M; row++) {
            for (col = 0; col < M; col++) {
                printf("%5.2f%s", A[row*M+col], (col < M-1) ? ", " : ";\n\t");
            }
        }
        printf("\nB = [");
        for (col = 0; col < M; col++) {
            printf("%5.2f%s", B[col], (col < M-1) ? "; " : "]\n");
        }
    }
}

void print_X() {
    int row;

    if (M < 10) {
        printf("\nX = [");
        for (row = 0; row < M; row++) {
            printf("%5.2f%s", x[row], (row < M-1) ? "; " : "]\n");
        }
    }
}



void fatal(char *message)
{
    printf("%s\n",message);
    exit(1);
}


void E_Finalize(float *a,float *b,float *x,float *f)
{
    free(a);
    free(b);
    free(x);
    free(f);
}


int main(int argc, char **argv)
{
    int i,j,t,k,my_rank,group_size;
    int inter;
    int i1,i2;
    int v,w;
    float temp;
    int tem;
    float *sum;
    float *f;
    float lmax;
    float *a;
    float *b;
    //float *x;
    int *shift;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    if(argc==1)
    {
    	if(my_rank==0)
    		printf("Enter the number of dimensions as parameter\n");
	      MPI_Finalize();
	      exit(1);
   }
//    FILE *fdA,*fdB;
    M = atoi(argv[1]);
    if(M%group_size!=0)
    {
    	if(my_rank==0)
    	    	printf("Enter the number of dimensions as the multiple of number of the processors\n");
    		    MPI_Finalize();
    		    exit(1);
    }
    A=(float *)malloc(floatsize*M*M);
    B=(float *)malloc(floatsize*M);
    x=(float *)malloc(floatsize*M);

    /* Initialize A and B */
    initialize_inputs();

    /* Print input matrices */




    p=group_size;

    starttime=MPI_Wtime();

    if(my_rank==0)
    {
    	for(i=1;i<group_size;i++)
    	{
    	MPI_Send(&M,1,MPI_INT,i,100,MPI_COMM_WORLD);
    	}
    	 }
    else
        	{
        		MPI_Recv(&M,1,MPI_INT,0,100,MPI_COMM_WORLD,&status);
        	}
    //MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);
    m=M/p;
    if (M%p!=0) m++;

    f=(float*)malloc(sizeof(float)*(M+1));
    a=(float*)malloc(sizeof(float)*m*M);
    b=(float*)malloc(sizeof(float)*m);
    sum=(float*)malloc(sizeof(float)*m);
    //x=(float*)malloc(sizeof(float)*M);
    shift=(int*)malloc(sizeof(int)*M);

    if (a==NULL||b==NULL||f==NULL||sum==NULL||x==NULL||shift==NULL)
        fatal("allocate error\n");

    for(i=0;i<M;i++)
        shift[i]=i;


    if (my_rank==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<M;j++)
                a(i,j)=A(i*p,j);

        for(i=0;i<m;i++)
            b(i)=B(i*p);
    }

    if (my_rank==0)
    {
        for(i=0;i<M;i++)
            if ((i%p)!=0)
        {
            i1=i%p;
            i2=i/p+1;

            MPI_Send(&A(i,0),M,MPI_FLOAT,i1,i2,MPI_COMM_WORLD);
            MPI_Send(&B(i),1,MPI_FLOAT,i1,i2,MPI_COMM_WORLD);
        }
    }
    else
    {
        for(i=0;i<m;i++)
        {
            MPI_Recv(&a(i,0),M,MPI_FLOAT,0,i+1,MPI_COMM_WORLD,&status);
            MPI_Recv(&b(i),1,MPI_FLOAT,0,i+1,MPI_COMM_WORLD,&status);
        }
    }

    time1=MPI_Wtime();

    for(i=0;i<m;i++)
        for(j=0;j<p;j++)
    {
        if (my_rank==j)
        {
            v=i*p+j;
            lmax=a(i,v);
            l=v;

            for(k=v+1;k<M;k++)
                if (fabs(a(i,k))>lmax)
            {
                lmax=a(i,k);
                l=k;
            }

            if (l!=v)
            {
                for(t=0;t<m;t++)
                {
                    temp=a(t,v);
                    a(t,v)=a(t,l);
                    a(t,l)=temp;
                }

                tem=shift[v];
                shift[v]=shift[l];
                shift[l]=tem;
            }

            for(k=v+1;k<M;k++)
                a(i,k)=a(i,k)/a(i,v);

            b(i)=b(i)/a(i,v);
            a(i,v)=1;

            for(k=v+1;k<M;k++)
                f[k]=a(i,k);
            f[M]=b(i);
for(inter=0;inter<group_size;inter++)
{
	if(my_rank!=inter)
	{
MPI_Send(&f[0],M+1,MPI_FLOAT,inter,2,MPI_COMM_WORLD);
MPI_Send(&l,1,MPI_INT,inter,4,MPI_COMM_WORLD);
	}
}

//MPI_Bcast(&f[0],M+1,MPI_FLOAT,my_rank,MPI_COMM_WORLD);
 //MPI_Bcast(&l,1,MPI_INT,my_rank,MPI_COMM_WORLD);
        }
        else
        {
            v=i*p+j;
           MPI_Recv(&f[0],M+1,MPI_FLOAT,MPI_ANY_SOURCE,2,MPI_COMM_WORLD,&status);
           MPI_Recv(&l,1,MPI_INT,MPI_ANY_SOURCE,4,MPI_COMM_WORLD,&status);
           // printf("send2:%d",inter);
            //MPI_Bcast(&f[0],M+1,MPI_FLOAT,j,MPI_COMM_WORLD);
           // MPI_Bcast(&l,1,MPI_INT,j,MPI_COMM_WORLD);

            if (l!=v)
            {
                for(t=0;t<m;t++)
                {
                    temp=a(t,v);
                    a(t,v)=a(t,l);
                    a(t,l)=temp;
                }

                tem=shift[v];
                shift[v]=shift[l];
                shift[l]=tem;
            }
        }
        if (my_rank<=j)
            for(k=i+1;k<m;k++)
        {
            for(w=v+1;w<M;w++)
                a(k,w)=a(k,w)-f[w]*a(k,v);
            b(k)=b(k)-f[M]*a(k,v);
        }

        if (my_rank>j)
            for(k=i;k<m;k++)
        {
            for(w=v+1;w<M;w++)
                a(k,w)=a(k,w)-f[w]*a(k,v);
            b(k)=b(k)-f[M]*a(k,v);
        }
    }


    for(i=0;i<m;i++)
        sum[i]=0.0;

    for(i=m-1;i>=0;i--)
        for(j=p-1;j>=0;j--)
            if (my_rank==j)
            {
                x[i*p+j]=(b(i)-sum[i])/a(i,i*p+j);
                for(inter=0;inter<group_size;inter++)
                {
                	if(my_rank!=inter)
                	{
                MPI_Send(&x[i*p+j],1,MPI_FLOAT,inter,3,MPI_COMM_WORLD);
                //printf("send1:%d",inter);
                	}
                }

                //MPI_Bcast(&x[i*p+j],1,MPI_FLOAT,my_rank,MPI_COMM_WORLD);
                for(k=0;k<i;k++)
                    sum[k]=sum[k]+a(k,i*p+j)*x[i*p+j];
            }
            else
            {
         MPI_Recv(&x[i*p+j],1,MPI_FLOAT,MPI_ANY_SOURCE,3,MPI_COMM_WORLD,&status);
        //MPI_Bcast(&x[i*p+j],1,MPI_FLOAT,j,MPI_COMM_WORLD);

        if (my_rank>j)
            for(k=0;k<i;k++)
                sum[k]=sum[k]+a(k,i*p+j)*x[i*p+j];

        if (my_rank<j)
            for(k=0;k<=i;k++)
                sum[k]=sum[k]+a(k,i*p+j)*x[i*p+j];
    }

    if (my_rank!=0)
        for(i=0;i<m;i++)
            MPI_Send(&x[i*p+my_rank],1,MPI_FLOAT,0,i,MPI_COMM_WORLD);
    else
        for(i=1;i<p;i++)
            for(j=0;j<m;j++)
                MPI_Recv(&x[j*p+i],1,MPI_FLOAT,i,j,MPI_COMM_WORLD,&status);

    if (my_rank==0)
    {
    	if(M< 10)
    	{
    	print_inputs();
        printf("\nOutput of solution\n");
        for(k=0;k<M;k++)
        {
            for(i=0;i<M;i++)
            {
                if (shift[i]==k) printf("x[%d]=%f\n",k,x[i]);
            }
        }
    	}
    }

    time2=MPI_Wtime();

    if (my_rank==0)
    {
        printf("\n");
        printf("Parallel compute time = %f seconds\n",time2-time1);
    }

    MPI_Finalize();
    E_Finalize(a,b,x,f);
   return(0);
}
