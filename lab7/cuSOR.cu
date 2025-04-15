/*
   Minimal CUDA program, intended just to test ability
   to compile and run a CUDA program

     nvcc cuda_test_lab7.cu -o cuda_test_lab7

   You need to follow instructions provided elsewhere, such as in the
   "CUDA_and-SCC-for-EC527,pdf" file, to setup your environment where you can
   compile and run this.

   To understand the program, of course you should read the lecture notes
   (slides) that have "GPU" in the name.
*/

#include <cstdio>
#include <cstdlib>
#include <math.h>

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define NUM_THREADS_PER_BLOCK   256
#define NUM_BLOCKS         16
#define PRINT_TIME         1
#define SM_ARR_LEN        1024
#define TOL            1e-6
#define CPNS 3.0
#define N 32
#define Nt 64
#define ITER 2000
#define OMEGA 1.85f

#define IMUL(a, b) __mul24(a, b)


/* code for 3c */

// Kernel for single thread per output element
__global__ void SOR_multi_ta(float* data) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j = tx;
    int start_i = ty;
    int i = start_i;

    // Boundary check for the element
    if (i > 0 && i < Nt - 1 && j > 0 && j < Nt - 1) {
        int idx = i * Nt + j;

        // Perform SOR for multiple iterations
        for (int it = 0; it < 1; ++it) {
            float up    = data[(i - 1) * Nt + j];
            float down  = data[(i + 1) * Nt + j];
            float left  = data[i * Nt + (j - 1)];
            float right = data[i * Nt + (j + 1)];

            float avg = 0.25f * (up + down + left + right);
            data[idx] += OMEGA * (avg - data[idx]);
        }
    }
}

// 3b
__global__ void SOR_multi_with_iterations(float* data, int iterations) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j = tx;
    int start_i = ty;
    int i = start_i;

    // Boundary check for the element
    if (i > 0 && i < Nt - 1 && j > 0 && j < Nt - 1) {
        int idx = i * Nt + j;

        // Perform SOR for the specified number of iterations
        for (int it = 0; it < iterations; ++it) {
            float up    = data[(i - 1) * Nt + j];
            float down  = data[(i + 1) * Nt + j];
            float left  = data[i * Nt + (j - 1)];
            float right = data[i * Nt + (j + 1)];

            float avg = 0.25f * (up + down + left + right);
            data[idx] += OMEGA * (avg - data[idx]);
        }
    }
}

// Function for CPU SOR (single thread per output element)
void sor_host_ta(float* data) {
    for (int it = 0; it < ITER; ++it) {
        for (int i = 1; i < Nt - 1; ++i) {
            for (int j = 1; j < Nt - 1; ++j) {
                float up    = data[(i - 1) * Nt + j];
                float down  = data[(i + 1) * Nt + j];
                float left  = data[i * Nt + (j - 1)];
                float right = data[i * Nt + (j + 1)];

                float avg = 0.25f * (up + down + left + right);
                data[i * Nt + j] += OMEGA * (avg - data[i * Nt + j]);
            }
        }
    }
}
void initializeArray1D(float *arr, int len, int seed);
void compare_results(float* cpu, float* gpu) {
    float max_diff = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float diff = fabs(cpu[i] - gpu[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    printf("Max absolute difference: %e\n", max_diff);
}
__global__ void kernel_add (int arrLen, float* x, float* y, float* result) {
  const int tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
  const int threadN = IMUL(blockDim.x, gridDim.x);

  int i;

  for(i = tid; i < arrLen; i += threadN) {
    result[i] = (1e-6 * x[i] ) + (1e-7 * y[i]) + 0.25;
  }
}

__global__ void SOR_multi(float* data) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int i;
  int j = tx;
  int start_i = ty * 4;

  for(int offset = 0; offset < 4; offset++)
  {
    i = start_i + offset;
    if(i > 0 && i < Nt-1 && j > 0 && j < Nt -1)
    {
    int idx = i * Nt + j;
    for (int it = 0; it < ITER; ++it) {
                float up    = data[(i - 1) * Nt + j];
                float down  = data[(i + 1) * Nt + j];
                float left  = data[i * Nt + (j - 1)];
                float right = data[i * Nt + (j + 1)];

                float avg = 0.25f * (up + down + left + right);
                data[idx] += OMEGA * (avg - data[idx]);
    }
    }
  }
}

void SOR_multi_host(float* data)
{
  for (int it = 0; it < ITER; ++it) {
        for (int i = 1; i < Nt - 1; ++i) {
            for (int j = 1; j < Nt - 1; ++j) {
                float up    = data[(i - 1) * Nt + j];
                float down  = data[(i + 1) * Nt + j];
                float left  = data[i * Nt + (j - 1)];
                float right = data[i * Nt + (j + 1)];

                float avg = 0.25f * (up + down + left + right);
                data[i * Nt + j] += OMEGA * (avg - data[i * Nt + j]);
            }
        }
    }
}

void print_diff(float* cpu, float* gpu){
  float max_diff = 0.0f;
    for (int i = 0; i < Nt * Nt; ++i) {
        float diff = fabs(cpu[i] - gpu[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    printf("difference for the multi version is %10.4g\n", max_diff);
}

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}


__global__ void SOR (float* data) {
  int i = threadIdx.x;
  int j = threadIdx.y;
  int index = i * N + j;

  // do not edit boundary
  if(i > 0 && i < N-1 && j > 0 && j < N-1) {
     for(int it = 0; it < ITER; it++)
     {
	float up = data[(i-1) * N + j];
	float down  = data[(i + 1) * N + j];
	float left  = data[i * N + (j - 1)];
	float right  = data[i * N + (j + 1)];

	float average = 0.235 * (up + down + left + right);
	data[index] += OMEGA * (average - data[index]);
     }
  }
}

int main(int argc, char **argv){
  int arrLen = 0;

  // GPU Timing variables
  cudaEvent_t start, stop;
  float elapsed_gpu;

  // Arrays on GPU global memoryc
  float *d_x;
  float *d_y;
  float *d_result;

  // Arrays on the host memory
  float *h_x;
  float *h_y;
  float *h_result;
  float *h_result_gold;

  int i, errCount = 0, zeroCount = 0;

  if (argc > 1) {
    arrLen  = atoi(argv[1]);
  }
  else {
    arrLen = SM_ARR_LEN;
  }

  printf("Length of the array = %d\n", arrLen);

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));
  struct timespec time_start, time_stop;
  // Allocate GPU memory
  size_t allocSize = arrLen * sizeof(float);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, allocSize));

  // Allocate arrays on host memory
  h_x                        = (float *) malloc(allocSize);
  h_y                        = (float *) malloc(allocSize);
  h_result                   = (float *) malloc(allocSize);
  h_result_gold              = (float *) malloc(allocSize);

  // Initialize the host arrays
  printf("\nInitializing the arrays ...");
  // Arrays are initialized with a known seed for reproducability
  initializeArray1D(h_x, arrLen, 2453);
  initializeArray1D(h_y, arrLen, 2453);
  printf("\t... done\n\n");

#if PRINT_TIME
  // Create the cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record event on the default stream
  cudaEventRecord(start, 0);
#endif

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, allocSize, cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(1, 1);
  // Launch the kernel
  SOR<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_x);

  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, allocSize, cudaMemcpyDeviceToHost));

#if PRINT_TIME
  // Stop and destroy the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_gpu, start, stop);
  printf("\nGPU time: %f (msec)\n", elapsed_gpu);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
#endif

  // Compute the results on the host
  float *temp = (float *)malloc(1024*sizeof(float));
  for(int i = 0; i < 1024; i++)
  {
    h_result_gold[i] = h_x[i];
  }
  // time the CPU version
  clock_gettime(CLOCK_REALTIME, &time_start);
  for(int iter = 0; iter < ITER; iter++)
  {
    for(int i = 0; i < 1024; i++)
    {
      temp[i] = h_result_gold[i];
    }

    for(int y = 1; y < 32-1; y++) {
      for(int x = 1; x < 32-1; x++)
      {
        int idx = y * 32 + x;
	int up = (y-1)*32+x;
	int down = (y+1)*32+x;
	int left = y*32+(x-1);
	int right = y*32+(x+1);
	h_result_gold[idx] = 0.25f * (temp[up] + temp[down] + temp[left] + temp[right]);
      }
    }
  }
  clock_gettime(CLOCK_REALTIME, &time_stop);
  printf("time from CPU is %10.4g\n", interval(time_start, time_stop));
  /*
  for(i = 0; i < 50; i++) {
    printf("%d:\t%.8f\t%.8f\n", i, h_result_gold[i], h_result[i]);
  }
  */
  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(d_x)); 

  free(h_x);
  free(h_y);
  free(h_result);


  // do it again for 2b
  int arrLen_s = Nt;
    // GPU Timing variables
  cudaEvent_t start_s, stop_s;
  float elapsed_gpu_s;

  // Arrays on GPU global memoryc
  float *d_x_s;
  float *d_y_s;
  float *d_result_s;

  // Arrays on the host memory
  float *h_x_s;
  float *h_y_s;
  float *h_result_s;
  float *h_result_gold_s;

  printf("Length of the array = %d\n", arrLen_s);

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

  // Allocate GPU memory
  size_t allocSize_s = arrLen_s*arrLen_s * sizeof(float);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_x_s, allocSize_s));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_y_s, allocSize_s));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_result_s, allocSize_s));

  // Allocate arrays on host memory
  h_x_s                        = (float *) malloc(allocSize_s);
  h_y_s                        = (float *) malloc(allocSize_s);
  h_result_s                   = (float *) malloc(allocSize_s);
  h_result_gold_s              = (float *) malloc(allocSize_s);

  // Initialize the host arrays
  printf("\nInitializing the arrays ...");
  // Arrays are initialized with a known seed for reproducability
  initializeArray1D(h_x_s, arrLen_s*arrLen_s, 2453);
  initializeArray1D(h_y_s, arrLen_s*arrLen_s, 2453);
  printf("\t... done\n\n");
  
  //cpu sor 64
  struct timespec time_start_s, time_stop_s;
  double time_stamp_s;
  clock_gettime(CLOCK_REALTIME, &time_start_s);
  SOR_multi_host(h_y_s);  
  clock_gettime(CLOCK_REALTIME, &time_stop_s);
  time_stamp_s = interval(time_start_s, time_stop_s);
  printf("\nCPU time: %f\n", time_stamp_s);
#if PRINT_TIME
  // Create the cuda events
  cudaEventCreate(&start_s);
  cudaEventCreate(&stop_s);
  // Record event on the default stream
  cudaEventRecord(start_s, 0);
#endif

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(d_x_s, h_x_s, allocSize_s, cudaMemcpyHostToDevice));

  // Launch the kernel
  dim3 threadsPerBlock_s(32, 32);
  SOR_multi<<<1, threadsPerBlock_s>>>(d_x_s);

  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(h_result_s, d_x_s, allocSize_s, cudaMemcpyDeviceToHost));

#if PRINT_TIME
  // Stop and destroy the timer
  cudaEventRecord(stop_s,0);
  cudaEventSynchronize(stop_s);
  cudaEventElapsedTime(&elapsed_gpu_s, start_s, stop_s);
  printf("\nGPU time: %f (msec)\n", elapsed_gpu_s);
  cudaEventDestroy(start_s);
  cudaEventDestroy(stop_s);
#endif

  compare_results(h_y_s, h_result_s);
  

  /* 3a */
  allocSize = Nt*Nt * sizeof(float);
  float * h_data_gpu = (float*)malloc(allocSize);
  float* h_data_cpu = (float*)malloc(allocSize);
  // Initialize the host arrays
  printf("\nInitializing the arrays ...");
  // Arrays are initialized with a known seed for reproducability 
  initializeArray1D(h_data_gpu, arrLen_s*arrLen_s, 2453);
  initializeArray1D(h_data_cpu, arrLen_s*arrLen_s, 2453);
  printf("\t... done\n\n");
  clock_gettime(CLOCK_REALTIME, &time_start_s);
  sor_host_ta(h_data_cpu);
  clock_gettime(CLOCK_REALTIME, &time_stop_s);
  printf("3a, cpu time: %f\n", interval(time_start_s, time_stop_s));
  // gpu
  printf("3A, block size of 32 x 32, 4x4\n");
  float* d_data;
  cudaMalloc((void**)&d_data, allocSize);

  cudaEvent_t start_ta, stop_ta;
  cudaEventCreate(&start_ta);
  cudaEventCreate(&stop_ta);
  cudaEventRecord(start_ta, 0);
  cudaMemcpy(d_data, h_data_gpu, allocSize, cudaMemcpyHostToDevice);

  // grid and block size
  dim3 tpb_sixteen(32, 32);
  dim3 bpg_sixteen(4, 4);
  for(int e = 0; e < ITER; e++)
  {
    SOR_multi_ta<<<bpg_sixteen, tpb_sixteen>>>(d_data);
    cudaDeviceSynchronize();
  }
  SOR_multi_ta<<<bpg_sixteen, tpb_sixteen>>>(d_data);
  cudaMemcpy(h_data_gpu, d_data, allocSize, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop_ta, 0);
  cudaEventSynchronize(stop_ta);
  cudaEventElapsedTime(&elapsed_gpu, start_ta, stop_ta);
  printf("\nGPU time: %f (msec)\n", elapsed_gpu);
  cudaFree(d_data);

  compare_results(h_data_cpu, h_data_gpu);
  /* 3b */
  // reinit
  initializeArray1D(h_data_gpu, arrLen_s*arrLen_s, 2453); 
  initializeArray1D(h_data_cpu, arrLen_s*arrLen_s, 2453);

  clock_gettime(CLOCK_REALTIME, &time_start_s);
  sor_host_ta(h_data_cpu);
  clock_gettime(CLOCK_REALTIME, &time_stop_s);
  printf("3B, 16x16 16x16\n");
  printf("3b, cpu time: %f\n", interval(time_start_s, time_stop_s));

  //initializeArray1D(h_data_cpu, arrLen_s*arrLen_s, 2453);
  float* d_datas;
  cudaMalloc((void**)&d_datas, allocSize);

  cudaEvent_t start_tas, stop_tas;
  cudaEventCreate(&start_tas);
  cudaEventCreate(&stop_tas);
  cudaEventRecord(start_tas, 0);
  cudaMemcpy(d_datas, h_data_gpu, allocSize, cudaMemcpyHostToDevice);

  // 32x32 threads in block
  dim3 threadspblock(16, 16);
  dim3 blockspgrid(16, 16);
  
  SOR_multi_with_iterations<<<blockspgrid, threadspblock>>>(d_datas, ITER);
  cudaDeviceSynchronize();

  cudaMemcpy(h_data_gpu, d_datas, allocSize, cudaMemcpyDeviceToHost);
  compare_results(h_data_cpu, h_data_gpu);
  cudaEventRecord(stop_tas, 0);
  cudaEventSynchronize(stop_tas);
  cudaEventElapsedTime(&elapsed_gpu, start_tas, stop_tas);
  printf("\nGPU time: %f (msec)\n", elapsed_gpu);
 
  free(h_data_gpu);
  free(h_data_cpu);

  return 0;
}

void initializeArray1D(float *arr, int len, int seed) {
  int i;
  float randNum;
  srand(seed);

  for (i = 0; i < len; i++) {
    randNum = (float) rand();
    arr[i] = randNum;
  }
}
