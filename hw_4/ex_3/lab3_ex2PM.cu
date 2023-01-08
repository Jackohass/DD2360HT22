
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
  //@@ Insert code to implement matrix multiplication here
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

//@@ Insert code to implement timer start
double startTime() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  currTime = ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
  return currTime;
}

//@@ Insert code to implement timer stop
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
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  FILE *fp = fopen("data/ex2OutputFloat.csv", "a");
  time_t t;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
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
  
  //@@ Insert code below to allocate Host memory for input and output
  cudaDeviceSynchronize();
  cudaHostAlloc((void**) &hostA, ASize, cudaHostAllocDefault);
  cudaHostAlloc((void**) &hostB, BSize, cudaHostAllocDefault);
  cudaHostAlloc((void**) &hostC, CSize, cudaHostAllocDefault);
  cudaHostAlloc((void**) &resultRef, CSize, cudaHostAllocDefault);
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  srand((unsigned) time(&t));
  for(int i = 0; i < numAColumns*numARows; i++)
  {
    hostA[i] = (DataType)rand() / RAND_MAX;
  }
  for(int i = 0; i < numBColumns*numBRows; i++)
  {
    hostB[i] = (DataType)rand() / RAND_MAX;
  }
  for (size_t i = 0; i < numCRows; i++)
  {
    for (size_t j = 0; j < numCColumns; j++)
    {
      DataType sum = 0;
      for (size_t k = 0; k < numAColumns; k++)
      {
        sum += hostA[k + i*numAColumns] * hostB[j + k*numBColumns];
      }
      resultRef[j+i*numCColumns] = sum;
    }
  }
  //Print out matrices
  /*printMatrix(hostA, numAColumns, numARows);
  printMatrix(hostB, numBColumns, numBRows);
  printMatrix(resultRef, numCColumns, numCRows);*/

  //Initalize Cuda
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device : %s\n", prop.name);
  cudaSetDevice(0);

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, ASize);
  cudaMalloc(&deviceB, BSize);
  cudaMalloc(&deviceC, CSize);

  //@@ Insert code to below to Copy memory to the GPU here
  startTime();
  cudaMemcpy(deviceA, hostA, ASize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, BSize, cudaMemcpyHostToDevice);
  double HToDTime = stopTime()*1000;

  //@@ Initialize the grid and block dimensions here
  dim3 block(32,32);
  dim3 grid
  (
    (int)ceil(((double)numCColumns)/((double)block.x)),
    (int)ceil(((double)numCRows)/((double)block.y))
  );

  //@@ Launch the GPU Kernel here
  startTime();
  gemm<<<grid, block>>>(deviceA, deviceB, deviceC, numARows, 
    numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  double kernelTime = stopTime()*1000;

  //@@ Copy the GPU memory back to the CPU here
  startTime();
  cudaMemcpy(hostC, deviceC, CSize, cudaMemcpyDeviceToHost);
  double DToHTime = stopTime()*1000;

  //@@ Insert code below to compare the output with the reference
  bool isAllCorrect = true;
  for(int i = 0; i < numCColumns*numCRows; i++)
  {
    DataType val = fabs(resultRef[i]-hostC[i]);
    //printf("Index: (%d, %d), Ref: %f, GPU: %f, Diff: %f\n", i%numCColumns,
    //    i/numCColumns, resultRef[i], hostC[i], val);
    if(val >= EPSILON)
    {
      //printf("%d: %f\n", i, val);
      printf("Index: (%d, %d), Ref: %f, GPU: %f, Diff: %f\n", i%numCColumns,
        i/numCColumns, resultRef[i], hostC[i], val);
      isAllCorrect = false;
    }
  }

  //Print GPU results
  //printMatrix(hostC, numCColumns, numCRows);
  
  if(isAllCorrect) printf("All correct!\n");
  else printf("Not all correct!\n");

  printf("H->D: %f ms, D->H: %f ms, Kernel: %f ms", HToDTime, DToHTime, kernelTime);
  //fprintf(fp, "(%d x %d) (%d x %d), %f, %f, %f\n", numARows, numAColumns, numBRows, numBColumns, HToDTime, DToHTime, kernelTime);

  //@@ Free the GPU memory here
  cudaFree(&deviceA);
  cudaFree(&deviceB);
  cudaFree(&deviceC);

  //@@ Free the CPU memory here
  cudaFreeHost(hostA);
  cudaFreeHost(hostB);
  cudaFreeHost(hostC);
  cudaFreeHost(resultRef);

  return 0;
}
