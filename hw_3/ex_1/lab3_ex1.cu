
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define DataType double
#define BLOCK_SIZE 1024
#define EPSILON 0.0000001

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = in1[i] + in2[i];
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

int main(int argc, char **argv)
{
  
  time_t t;
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  FILE *fp = fopen("data/ex1Output.csv", "a");

  //@@ Insert code below to read in inputLength from args
  if(argc < 2)
  {
    printf("Give me N");
    return 0;
  }
  inputLength = atoi(argv[1]);
  size_t size = inputLength*sizeof(DataType);
  
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType*)malloc(size);
  hostInput2 = (DataType*)malloc(size);
  hostOutput = (DataType*)malloc(size);
  resultRef = (DataType*)malloc(size);
  
  //Initalize Cuda
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device : %s\n", prop.name);
  cudaSetDevice(0);
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand((unsigned) time(&t));
  for(int i = 0; i < inputLength; i++)
  {
    hostInput1[i] = (DataType)rand() / RAND_MAX;
    hostInput2[i] = (DataType)rand() / RAND_MAX;
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, size);
  cudaMalloc(&deviceInput2, size);
  cudaMalloc(&deviceOutput, size);

  //@@ Insert code to below to Copy memory to the GPU here
  startTime();
  cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
  double HToDTime = stopTime()*1000;

  //@@ Initialize the 1D grid and block dimensions here
  dim3 block(BLOCK_SIZE);
  //dim3 grid(inputLength/BLOCK_SIZE);
  dim3 grid((int)ceil(((double)inputLength)/((double)BLOCK_SIZE)));

  //@@ Launch the GPU Kernel here
  startTime();
  vecAdd<<<grid, block>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  double kernelTime = stopTime()*1000;

  //@@ Copy the GPU memory back to the CPU here
  startTime();
  cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
  double DToHTime = stopTime()*1000;

  //@@ Insert code below to compare the output with the reference
  bool isAllCorrect = true;
  for(int i = 0; i < inputLength; i++)
  {
    DataType val = resultRef[i]-hostOutput[i];
    //printf("Ref: %f, GPU: %f, Diff: %f\n", resultRef[i], hostOutput[i], val);
    if(val >= EPSILON)
    {
      //printf("%d: %f\n", i, val);
      printf("Index: %d, Ref: %f, GPU: %f, Diff: %f\n", i, resultRef[i], hostOutput[i], val);
      isAllCorrect = false;
    }
  }
  if(isAllCorrect) printf("All correct!\n");
  else printf("Not all correct!\n");

  printf("H->D: %f ms, D->H: %f ms, Kernel: %f ms", HToDTime, DToHTime, kernelTime);
  fprintf(fp, "%d, %f, %f, %f\n", inputLength, HToDTime, DToHTime, kernelTime);

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}
