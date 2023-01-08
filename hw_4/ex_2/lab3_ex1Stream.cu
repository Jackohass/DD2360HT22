
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define DataType double
#define BLOCK_SIZE 1024
#define EPSILON 0.0000001

__global__ void vecAdd(DataType *ins, DataType *out, int len)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = ins[i*2] + ins[i*2+1];
  else return;
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
  int S_seg;
  DataType *hostInputs;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInputs;
  DataType *deviceOutput;

  FILE *fp = fopen("data/ex1Output.csv", "a");

  //@@ Insert code below to read in inputLength from args
  if(argc < 3)
  {
    printf("Give me N and S_seg");
    return 0;
  }
  inputLength = atoi(argv[1]);
  S_seg = atoi(argv[2]);

  size_t size = inputLength*sizeof(DataType);
  size_t segSize = S_seg*sizeof(DataType);
  int numSegs = inputLength/S_seg;
  
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  cudaHostAlloc((void**) &hostInputs, size*2, cudaHostAllocDefault);
  cudaHostAlloc((void**) &hostOutput, size, cudaHostAllocDefault);
  cudaHostAlloc((void**) &resultRef, size, cudaHostAllocDefault);
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand((unsigned) time(&t));
  for(int i = 0; i < inputLength; i += 2)
  {
    hostInputs[i] = (DataType)rand() / RAND_MAX;
    hostInputs[1 + i] = (DataType)rand() / RAND_MAX;
    resultRef[i/2] = hostInputs[i] + hostInputs[i+1];
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInputs, size*2);
  cudaMalloc(&deviceOutput, size);
  
  //Create streams
  const int numStreams = 4;
  cudaStream_t streams[numStreams];
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);
  cudaStreamCreate(&streams[2]);
  cudaStreamCreate(&streams[3]);

  //@@ Initialize the 1D grid and block dimensions here
  dim3 block(S_seg > BLOCK_SIZE ? BLOCK_SIZE : S_seg);
  dim3 grid((int)ceil(((double)S_seg)/((double)block.x)));

  //@@ Launch the GPU Kernel here
  startTime();
  int i = 0;
  for(; i < numSegs-1; i++)
  {
    int offset = i * S_seg;
    cudaMemcpyAsync(&deviceInputs[offset*2], &hostInputs[offset*2], segSize*2, cudaMemcpyHostToDevice, streams[i%numStreams]);
    vecAdd<<<grid, block, 0, streams[i%numStreams]>>>(&deviceInputs[offset*2], &deviceOutput[offset], S_seg);
    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], segSize, cudaMemcpyDeviceToHost, streams[i%numStreams]);
  }
  int offset = i * S_seg;
  cudaMemcpyAsync(&deviceInputs[offset*2], &hostInputs[offset*2], size*2 - i*segSize*2, cudaMemcpyHostToDevice, streams[i%numStreams]);
  vecAdd<<<grid, block, 0, streams[i%numStreams]>>>(&deviceInputs[offset*2], &deviceOutput[offset], inputLength-offset);
  cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], size - i*segSize, cudaMemcpyDeviceToHost, streams[i%numStreams]);

  cudaDeviceSynchronize();
  double totTime = stopTime()*1000;

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

  printf("Total time: %f", totTime);
  fprintf(fp, "%d, %d, %f\n", S_seg, inputLength, totTime);

  //Destroy streams
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
  cudaStreamDestroy(streams[2]);
  cudaStreamDestroy(streams[3]);

  //@@ Free the GPU memory here
  cudaFree(deviceInputs);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  cudaFreeHost(hostInputs);
  cudaFreeHost(hostOutput);
  cudaFreeHost(resultRef);

  return 0;
}
