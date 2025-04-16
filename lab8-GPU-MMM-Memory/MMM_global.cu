/*
   Minimal CUDA program, intended just to test ability
   to compile and run a CUDA program

     nvcc cuda_test_lab7.cu -o cuda_test_lab7

   You need to follow instructions provided elsewhere, such as in the
   "CUDA_and-SCC-for-EC527.pdf" file, to set up your environment where you can
   compile and run this.

   To understand the program, of course you should read the lecture notes
   (slides) that have "GPU" in the name.
*/

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

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
/* 
#define NUM_THREADS_PER_BLOCK   256
#define NUM_BLOCKS              16
#define PRINT_TIME              1
#define SM_ARR_LEN              50000
#define TOL                     1e-6
*/

#define TILE_SIZE 16
#define IMUL(a, b) __mul24(a, b)
#define IDX2R(i,j,N) ((i)*(N)+(j))

/* instantiate matrix */
void init_matrix(float* matrix, int N)
{
  // predictable values
  for(int i = 0; i < N*N; i++)
  {
    matrix[i] = (float)(rand() % 10);
  }
}

/* CPU MMM for validation */
void cpu_mmm(const float* A, const float* B, float* C, int N)
{
  for(int i = 0; i < N; i++)
  {
    for(int j = 0; j < N; j++)
    {
      float sum = 0.0f;
      for(int k = 0; k < N; k++)
      {
        sum += A[IDX2R(i, k, N)] * B[IDX2R(k, j, N)];
      }
      C[IDX2R(i, j, N)] = sum;
    }
  }
}

// MMM only using global memory
__global__ void mmm_global(float* A, float*B, float* C, int arrLen)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < arrLen && col < arrLen)
  {
    float sum = 0.0f;
    for(int k = 0; k < arrLen; k++)
    {
      sum += A[IDX2R(row, k, arrLen)] * B[IDX2R(k, col, arrLen)];
    }
    C[IDX2R(row, col, arrLen)] = sum;
  }
}

// MMM using shared memory
__global__ void mmm_shared(float* A, float* B, float* C, int arrLen)
{
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];
  
  int tidy = threadIdx.y;
  int tidx = threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int limit = arrLen / TILE_SIZE;

  float sum = 0.0f;
  for(int t = 0; t < limit; t++)
  {
    tileA[tidy][tidx] = A[row * arrLen + (t * TILE_SIZE + tidx)];
    tileB[tidy][tidx] = B[(t * TILE_SIZE + tidy) * arrLen + col];
    __syncthreads();

    for(int k = 0; k < TILE_SIZE; k++)
    {
      sum += tileA[tidy][k] * tileB[k][tidx];
    }
    __syncthreads();
  }
  
  if(row < arrLen && col < arrLen)
  {
    C[row * arrLen + col] = sum;
  }
}

// MMM with shared memory and loop unrolling
__global__ void mmm_shared_unroll(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tidy = threadIdx.y;
    int tidx = threadIdx.x;

    int row = blockIdx.y * TILE_SIZE + tidy;
    int col = blockIdx.x * TILE_SIZE + tidx;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE_SIZE; t++) {
        tileA[tidy][tidx] = A[row * N + (t * TILE_SIZE + tidx)];
        tileB[tidy][tidx] = B[(t * TILE_SIZE + tidy) * N + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[tidy][k] * tileB[k][tidx];
        __syncthreads();
    }

    if (row < N && col < N)
        C[IDX2R(row, col, N)] = sum;
}

/* non-coalesced global */
__global__ void mmm_bad_global(float* A, float* B, float* C, int N)
{
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;  // flipping this index

  if(row < N && col < N)
  {
    float sum = 0.0f;
    for(int k = 0; k < N; k++)
    {
      sum += A[IDX2R(row, k, N)] * B[IDX2R(k, col, N)];
    }
    C[IDX2R(row, col, N)] = sum;
 }
}

// shared memory bank conflict
__global__ void mmm_shared_conflict(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1]; // misalign shared memory
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];
    
    int tidy = threadIdx.y;
    int tidx = threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + tidy;
    int col = blockIdx.x * TILE_SIZE + tidx;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE_SIZE; t++) {
        tileA[tidy][tidx] = A[row * N + (t * TILE_SIZE + tidx)];
        tileB[tidy][tidx] = B[(t * TILE_SIZE + tidy) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++){
            sum += tileA[tidy][k] * tileB[k][tidx];
        }
        __syncthreads();
    }

    if (row < N && col < N){
        C[IDX2R(row, col, N)] = sum;
    }
}

__global__ void mmm_global_shared_bad(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1]; // +1 forces shared memory bank conflict
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];

    int col = blockIdx.y * TILE_SIZE + threadIdx.y;  // flipped index access (bad for global)
    int row = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE_SIZE; t++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// validate, getting the max tolerance
void validate(const float* ref, const float* test, int N)
{
  float max_error = 0.0f;
  for(int i = 0; i < N * N; i++)
  {
    max_error += fabsf(ref[i] - test[i]);;
  }
  printf("MAX ERROR: %f\n", max_error);
}

/* funciton that launches all of the different MMMs */
void run_gpu_mmm(const char* label, int N,
                 void (*kernel)(float*, float*, float*, int),
                 dim3 blockDim) {
    size_t bytes = N * N * sizeof(float);
// host memory allocation
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_ref = (float*)malloc(bytes);

    init_matrix(h_A, N);
    init_matrix(h_B, N);

    // CPU baseline for validation
    clock_t cpu_start = clock();
    cpu_mmm(h_A, h_B, h_C_ref, N);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("[CPU] Time: %.8f s\n", cpu_time);
// device alloc
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
// event setup
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);

    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop); 
// start total timer
    cudaEventRecord(total_start);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
// grid configure
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

// kernel timer
    cudaEventRecord(kernel_start);
// launch kernel
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
// copy back results
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);

    float total_ms = 0;
    float kernel_ms = 0;
    cudaEventElapsedTime(&total_ms, total_start, total_stop);
    cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);
    printf("[%s] GPU total time (incl. memcpy): %.8f s\n", label, total_ms / 1000.0f);
    printf("[%s] GPU kernel time: %.8f s\n", label, kernel_ms / 1000.0f);
    validate(h_C_ref, h_C, N);

    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
/* 
__global__ void kernel_add (int arrLen, float* x, float* y, float* result) {
  const int tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
  const int threadN = IMUL(blockDim.x, gridDim.x);

  for (int i = tid; i < arrLen; i += threadN) {
    result[i] = (1e-6 * x[i]) + (1e-7 * y[i]) + 0.25;
  }
}
*/


int main(int argc, char **argv){
    int sizes[] = {16, 32, 64, 128, 256, 512, 1024, 2048};
    dim3 blockDim(TILE_SIZE, TILE_SIZE);

    for (int i = 0; i < 8; i++) {
        int N = sizes[i];
        printf("\n===== Matrix Size: %d x %d =====\n", N, N);

        run_gpu_mmm("Global Memory", N, mmm_global, blockDim);
        run_gpu_mmm("Shared Memory", N, mmm_shared, blockDim);
        run_gpu_mmm("Shared + Loop Unroll", N, mmm_shared_unroll, blockDim);
        run_gpu_mmm("Bad Global Access (4a)", N, mmm_bad_global, blockDim);
        run_gpu_mmm("Shared Conflict (4b)", N, mmm_shared_conflict, blockDim);
        run_gpu_mmm("Global + Shared Bad (4c)", N, mmm_global_shared_bad, blockDim);
    }
    return 0;
}

void initializeArray1D(float *arr, int len, int seed) {
  srand(seed);
  for (int i = 0; i < len; i++) {
    arr[i] = (float) rand();
  }
}

