#include <stdio.h>
#include <cuda.h>

#define INIT 1000
#define k 2

void random(int* x){
	for(int i=0;i<INIT*k;i++){
		x[i] = rand() % 10;
	}
}

__global__ void kernel(int *a, int *b, int *c){
	// //計算區塊索引
  // int block=(blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x;
  // //計算執行緒索引
  // int t=(threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
  // //計算區塊中包含的執行緒數目
  // int n=blockDim.x*blockDim.y*blockDim.z;
  // //執行緒在陣列中對應的位置
  // int x=block*n+t;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	c[x] = a[x] * b[x];
}

int main(void){
	int a[INIT*k] = {0};
	int b[INIT*k] = {0};
	int c[INIT*k] = {0};

	for(int i=0;i<INIT*k;i++){
		printf("%d: %d * %d = %d\n", i, a[i], b[i], c[i]);
	}
	printf("\n");

	int *GA, *GB, *GC;
	random(a);
	cudaMalloc((void**)&GA, k*INIT*sizeof(int));
	cudaMemcpy(GA, a, sizeof(int)*INIT*k, cudaMemcpyHostToDevice);

	random(b);
	cudaMalloc((void**)&GB, INIT*k*sizeof(int));
	cudaMemcpy(GB, b, sizeof(int)*INIT*k, cudaMemcpyHostToDevice);

	for(int i=0;i<INIT*k;i++){
		printf("%d: %d * %d = %d\n", i, a[i], b[i], c[i]);
	}
	printf("\n");

	cudaMalloc((void**)&GC, k*INIT*sizeof(int));

	kernel<<<k, INIT>>>(GA,GB,GC);
	cudaMemcpy(c, GC, sizeof(int)*INIT*k, cudaMemcpyDeviceToHost);

	for(int i=0;i<INIT*k;i++){
		printf("%d:\t %d * %d = %d\n", i, a[i], b[i], c[i]);
	}

	cudaFree(GA);
	cudaFree(GB);
	cudaFree(GC);

	return 0;
}
