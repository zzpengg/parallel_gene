#include <stdio.h>
#include <cuda.h>

#define INIT 1100
void random(int* x){
	for(int i=0;i<INIT;i++){
		x[i] = rand() % 10;
	}
}

__global__ void kernel(int *a, int *b, int *c){
	int x = blockIdx.x * threadIdx.x + threadIdx.x;
	c[x] = a[x] * b[x];
}

int main(void){
	int a[INIT] = {0};
	int b[INIT] = {0};
	int c[INIT] = {0};
	
	for(int i=0;i<INIT;i++){
		printf("%d * %d = %d\n", a[i], b[i], c[i]);
	}
	printf("\n");
	
	int *GA, *GB, *GC;
	random(a);
	cudaMalloc((void**)&GA, INIT*sizeof(int));
	cudaMemcpy(GA, a, sizeof(int)*INIT, cudaMemcpyHostToDevice);

	random(b);
	cudaMalloc((void**)&GB, INIT*sizeof(int));
	cudaMemcpy(GB, b, sizeof(int)*INIT, cudaMemcpyHostToDevice);
	
	for(int i=0;i<INIT;i++){
		printf("%d * %d = %d\n", a[i], b[i], c[i]);
	}
	printf("\n");
	
	cudaMalloc((void**)&GC, INIT*sizeof(int));
	
	kernel<<<1, INIT>>>(GA,GB,GC);
	cudaMemcpy(c, GC, sizeof(int)*INIT, cudaMemcpyDeviceToHost);
	
	for(int i=0;i<INIT;i++){
		printf("%d:\t %d * %d = %d\n", i, a[i], b[i], c[i]);
	}
	
	cudaFree(GA);
	cudaFree(GB);
	cudaFree(GC);
	
	return 0;
}
