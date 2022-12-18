
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define DataType unsigned int
#define NUM_BINS 4096
#define BINSIZE NUM_BINS*sizeof(DataType)
#define BLOCK_SIZE 1024
#define EPSILON 0

//Naivest solution
__global__ void histogram_kernelNaive(DataType *input, DataType *bins,
  DataType num_elements, DataType num_bins)
{
  //@@ Insert code below to compute histogram of input using shared memory and atomics
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_elements) return;

  const int s_idx = threadIdx.x;
  __shared__ DataType s_prod[NUM_BINS];
  for (int i = 0; i < 4; i++)
  {
    s_prod[s_idx*4 + i] = 0;
  }
  DataType num0 = input[idx];
  __syncthreads();

  atomicAdd(s_prod + num0, 1);
  __syncthreads();

  if (s_idx == 0)
  {
    for (int i = 0; i < NUM_BINS; i++)
    {
      int index = (i + blockIdx.x)%NUM_BINS;
      atomicAdd(bins + i, s_prod[i]);
    }
  }
}


//Focus on reading multiple elements and putting it into the array, but lowering the chance since the thread is divided up by 3.
/*__global__ void histogram_kernelN(DataType *input, DataType *bins,
  DataType num_elements, DataType num_bins)
{
  //@@ Insert code below to compute histogram of input using shared memory and atomics
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_elements) return;

  const int s_idx = threadIdx.x;
  __shared__ DataType s_prod[NUM_BINS*3];
  atomicAdd(s_prod + input[idx] + NUM_BINS*(s_idx % 3), 1);
  __syncthreads();

  if (s_idx == 0 || s_idx == 1 || s_idx == 2) {
    int blockSum = 0;
    for (int j = 0; j < blockDim.x; ++j) {
      blockSum += s_prod[j];
    }
    // Try each of two versions of adding to the accumulator
    if (ATOMIC) {
      atomicAdd(d_res, blockSum);
    } else {
      *d_res += blockSum;
    }
  }
}*/

//Focus on reading 3 elements and putting it into 3 different arrays.
__global__ void histogram_kernel3(DataType *input, DataType *bins,
  DataType num_elements, DataType num_bins)
{
  //@@ Insert code below to compute histogram of input using shared memory and atomics
  const int idx = threadIdx.x*3 + blockDim.x*3 * blockIdx.x;
  const int s_idx = threadIdx.x;
  __shared__ DataType s_prod[NUM_BINS*3];
  for (int i = 0; i < 4; i++)
  {
    s_prod[s_idx*4 + i] = 0;
    s_prod[s_idx*4 + i + NUM_BINS] = 0;
    s_prod[s_idx*4 + i + NUM_BINS*2] = 0;
  }
  DataType num0;
  DataType num1;
  DataType num2;

  if (idx < num_elements)
  {
    num0 = input[idx];
    num1 = input[idx + 1];
    num2 = input[idx + 2];
  }
    __syncthreads();
  if (idx < num_elements)
  {
    atomicAdd(s_prod + num0 + NUM_BINS*(s_idx%3), 1);
    atomicAdd(s_prod + num1 + NUM_BINS*((s_idx + 1)%3), 1);
    atomicAdd(s_prod + num2 + NUM_BINS*((s_idx + 2)%3), 1);
  }
  __syncthreads();
  
  for (int i = 0; i < 4; i++)
  {
    s_prod[s_idx*4 + i] += s_prod[s_idx*4 + i + NUM_BINS] + s_prod[s_idx*4 + i + NUM_BINS*2];
  }
  __syncthreads();

  if (s_idx == 0)
  {
    for (int i = 0; i < NUM_BINS; i++)
    {
      int index = (i + blockIdx.x)%NUM_BINS;
      atomicAdd(bins + i, s_prod[i]);
    }
  }
}

//Focus on reading lots and doing few atomic actions
/*__global__ void histogram_kernelR(DataType *input, DataType *bins,
  int num_elements, int num_bins)
{
  //@@ Insert code below to compute histogram of input using shared memory and atomics
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_elements) return;

  const int s_idx = threadIdx.x;
  __shared__ DataType s_prod[NUM_BINS*3];
  atomicAdd(s_prod + input[idx] + NUM_BINS*(s_idx % 3), 1);
  __syncthreads();

  if (s_idx == 0 || s_idx == 1 || s_idx == 2) {
    int blockSum = 0;
    for (int j = 0; j < blockDim.x; ++j) {
      blockSum += s_prod[j];
    }
    // Try each of two versions of adding to the accumulator
    if (ATOMIC) {
      atomicAdd(d_res, blockSum);
    } else {
      *d_res += blockSum;
    }
  }
}*/

__global__ void convert_kernel(DataType *bins, DataType num_bins) 
{
  //@@ Insert code below to clean up bins that saturate at 127
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_bins) return;
  if(bins[idx] > 127) bins[idx] = 127;
}

//@@ Insert code to implement timer start
double currTime;

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


int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput;
  DataType *hostBins;
  DataType *resultRef;
  DataType *deviceInput;
  DataType *deviceBins;

  time_t t;
  FILE *fp = fopen("data/ex3Output.csv", "a");
  FILE *histogram = fopen("data/histogram.csv", "w");

  //@@ Insert code below to read in inputLength from args
  if(argc < 2)
  {
    printf("Give me N");
    return 0;
  }
  inputLength = atoi(argv[1]);
  if(inputLength%3 != 0) inputLength += 3 - inputLength%3;
  size_t size = inputLength*sizeof(DataType);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (DataType*)malloc(size);
  hostBins = (DataType*)malloc(BINSIZE);
  resultRef = (DataType*)malloc(BINSIZE);
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  srand((unsigned) time(&t));
  for(int i = 0; i < inputLength; i++)
  {
    hostInput[i] = (DataType)(rand()%NUM_BINS);
  }

  //@@ Insert code below to create reference result in CPU
  startTime();
  for(int i = 0; i < NUM_BINS; i++)
  {
    resultRef[i] = 0;
  }
  for(int i = 0; i < inputLength; i++)
  {
    if(resultRef[hostInput[i]] < 127) resultRef[hostInput[i]] += 1;
  }
  double CPUTime = stopTime()*1000;

  //Initalize Cuda
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device : %s\n", prop.name);
  cudaSetDevice(0);

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, size);
  cudaMalloc(&deviceBins, BINSIZE);

  //@@ Insert code to Copy memory to the GPU here
  startTime();
  cudaMemcpy(deviceInput, hostInput, size, cudaMemcpyHostToDevice);
  double HToDTime = stopTime()*1000;

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, BINSIZE);

  //@@ Initialize the grid and block dimensions here
  dim3 block0(BLOCK_SIZE);
  dim3 grid0((int)ceil(((double)inputLength)/((double)BLOCK_SIZE*3)));
  //dim3 grid0((int)ceil(((double)inputLength)/((double)BLOCK_SIZE)));

  //@@ Launch the GPU Kernel here
  startTime();
  histogram_kernel3<<<grid0, block0>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  //histogram_kernelNaive<<<grid0, block0>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  //@@ Initialize the second grid and block dimensions here
  dim3 block1(BLOCK_SIZE);
  dim3 grid1(4);

  //@@ Launch the second GPU Kernel here
  cudaDeviceSynchronize();
  convert_kernel<<<grid1, block1>>>(deviceBins, NUM_BINS);
  cudaDeviceSynchronize();
  double kernelTime = stopTime()*1000;

  //@@ Copy the GPU memory back to the CPU here
  startTime();
  cudaMemcpy(hostBins, deviceBins, BINSIZE, cudaMemcpyDeviceToHost);
  double DToHTime = stopTime()*1000;

  //@@ Insert code below to compare the output with the reference
  bool isAllCorrect = true;
  for(int i = 0; i < NUM_BINS; i++)
  {
    DataType val = resultRef[i]-hostBins[i];
    //printf("Ref: %f, GPU: %f, Diff: %f\n", resultRef[i], hostOutput[i], val);
    if(val != EPSILON)
    {
      printf("Index: %d, Ref: %d, GPU: %d, Diff: %d\n", i, resultRef[i], hostBins[i], val);
      isAllCorrect = false;
    }
  }
  if(isAllCorrect) printf("All correct!\n");
  else printf("Not all correct!\n");

  printf("CPU: %f ms, H->D: %f ms, D->H: %f ms, Kernel: %f ms\n", CPUTime, HToDTime, DToHTime, kernelTime);
  fprintf(fp, "Size: %d, %f, %f, %f, %f\n", inputLength, CPUTime, HToDTime, DToHTime, kernelTime);
  
  fprintf(histogram, "Bin number, Bin size\n");
  for(int i = 0; i < NUM_BINS; i++)
  {
    fprintf(histogram, "%d, %d\n", i, hostBins[i]);
  }
  

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

