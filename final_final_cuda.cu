#include <unistd.h>
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define ISLAND 15
#define POPULATION 500
#define N ISLAND*POPULATION

#define MAX 100

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

  short b=blockIdx.x;       //區塊索引 == ISLAND
  short t=threadIdx.x;      //執行緒索引 == POPULATION
  short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  short x=b*n+t;

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[x]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, unsigned int* numbers) {
  /* curand works like rand - except that it takes a state as a parameter */

  short b=blockIdx.x;       //區塊索引 == ISLAND
  short t=threadIdx.x;      //執行緒索引 == POPULATION
  short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  short x=b*n+t;
  
  numbers[x] = curand(&states[x]) % 100;
}

int main( ) {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t* states;

  /* allocate space on the GPU for the random states */
  cudaMalloc((void**) &states, N * sizeof(curandState_t));

  /* invoke the GPU to initialize all of the random states */
  init<<<ISLAND, POPULATION>>>(time(0), states);

  /* allocate an array of unsigned ints on the CPU and GPU */
  unsigned int cpu_nums[N];
  unsigned int* gpu_nums;
  cudaMalloc((void**) &gpu_nums, N * sizeof(unsigned int));

  unsigned int cpu_nums2[N];
  unsigned int* gpu_nums2;
  cudaMalloc((void**) &gpu_nums2, N * sizeof(unsigned int));

  /* invoke the kernel to get some random numbers */
  randoms<<<ISLAND, POPULATION>>>(states, gpu_nums, gpu_nums2);

  /* copy the random numbers back */
  cudaMemcpy(cpu_nums, gpu_nums, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_nums2, gpu_nums2, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  /* print them out */
  for (int i = 0; i < N; i++) {
    printf("%u %u\n", cpu_nums[i], cpu_nums2[i]);
  }

  /* free the memory we allocated for the states and numbers */
  cudaFree(states);
  cudaFree(gpu_nums);
  cudaFree(gpu_nums2);

  return 0;
}