
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define DataType float
#define BLOCK_SIZE 1024
//#define EPSILON 0.0000001
#define EPSILON 0.0001

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
  int numAColumns, int numBRows, int numBColumns)
{
  //Matrix multiplication
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j >= numBColumns) return;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if(i >= numARows) return;
  
  DataType sum = 0;
  for (size_t k = 0; k < numAColumns; k++)
  {
    sum += A[k + i*numAColumns] * B[j + k*numBColumns];
  }
  C[j+i*numBColumns] = sum;
}

double currTime;

//Timer start
double startTime() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  currTime = ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
  return currTime;
}

//Timer stop
double stopTime() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6) - currTime;
}

void printMatrix(DataType* arr, int cols, int rows)
{
  printf("%d x %d\n", cols, rows);
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%f ", arr[j + i*cols]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv) 
{
  DataType *A; // The A matrix
  DataType *B; // The B matrix
  DataType *C; // The output C matrix

  DataType *resultRef; // The reference result

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  FILE *fp = fopen("data/ex2OutputFloat.csv", "a");
  time_t t;

  //Read in numARows, numAColumns, numBColumns from args
  if(argc < 4)
  {
    printf("Give me some args");
    return 0;
  }
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBColumns = atoi(argv[3]);

  numBRows = numAColumns;
  numCRows = numARows;
  numCColumns = numBColumns;

  size_t ASize = numAColumns*numARows*sizeof(DataType); 
  size_t BSize = numBColumns*numBRows*sizeof(DataType);
  size_t CSize = numCColumns*numCRows*sizeof(DataType);

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows,
    numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //Allocate memory for reference solution
  resultRef = (DataType*)malloc(CSize);

  //Allocate managed memory
  cudaDeviceSynchronize();
  cudaMallocManaged(&A, ASize);
  cudaMallocManaged(&B, BSize);
  cudaMallocManaged(&C, CSize);
  
  //Initialize hostA and hostB to random numbers, and create reference result in CPU
  srand((unsigned) time(&t));
  for(int i = 0; i < numAColumns*numARows; i++)
  {
    A[i] = (DataType)rand() / RAND_MAX;
  }
  for(int i = 0; i < numBColumns*numBRows; i++)
  {
    B[i] = (DataType)rand() / RAND_MAX;
  }
  
  //Create reference result in CPU
  for (size_t i = 0; i < numCRows; i++)
  {
    for (size_t j = 0; j < numCColumns; j++)
    {
      DataType sum = 0;
      for (size_t k = 0; k < numAColumns; k++)
      {
        sum += A[k + i*numAColumns] * B[j + k*numBColumns];
      }
      resultRef[j+i*numCColumns] = sum;
    }
  }
  //Print out matrices
  /*printMatrix(hostA, numAColumns, numARows);
  printMatrix(hostB, numBColumns, numBRows);
  printMatrix(resultRef, numCColumns, numCRows);*/

  //Initialize the grid and block dimensions
  dim3 block(32,32);
  dim3 grid
  (
    (int)ceil(((double)numCColumns)/((double)block.x)),
    (int)ceil(((double)numCRows)/((double)block.y))
  );

  //Launch the GPU Kernel
  startTime();
  gemm<<<grid, block>>>(A, B, C, numARows, 
    numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  double totTime = stopTime()*1000;

  //Compare the output with the reference
  bool isAllCorrect = true;
  for(int i = 0; i < numCColumns*numCRows; i++)
  {
    DataType val = fabs(resultRef[i]-C[i]);
    //printf("Index: (%d, %d), Ref: %f, GPU: %f, Diff: %f\n", i%numCColumns,
    //    i/numCColumns, resultRef[i], hostC[i], val);
    if(val >= EPSILON)
    {
      //printf("%d: %f\n", i, val);
      printf("Index: (%d, %d), Ref: %f, GPU: %f, Diff: %f\n", i%numCColumns,
        i/numCColumns, resultRef[i], C[i], val);
      isAllCorrect = false;
    }
  }

  //Print GPU results
  //printMatrix(hostC, numCColumns, numCRows);
  
  if(isAllCorrect) printf("All correct!\n");
  else printf("Not all correct!\n");

  printf("Total time: %f ms\n", totTime);
  //fprintf(fp, "(%d x %d) (%d x %d), %f\n", numARows, numAColumns, numBRows, numBColumns, totTime);

  //Free the managed memory
  cudaFree(&A);
  cudaFree(&B);
  cudaFree(&C);

  //Free the CPU memory
  free(resultRef);

  return 0;
}
