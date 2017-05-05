/*--------------------------------------------
* Date：2015-3-18
* Author：李根
* FileName：.cpp
* Description：CUDA二维数组加法
------------------------------------------------*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

static const int M = 4;
static const int N = 3;

//矩阵加法的kernel
__global__ void addMat(int **A,int **B,int **C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < M && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{int **A = (int **)malloc(M*sizeof(int *));  //host memory
    int **B = (int **)malloc(M*sizeof(int *));  //host memory
    int **C = (int **)malloc(M*sizeof(int *));  //host memory
    int *dataA =(int *)malloc(M*N*sizeof(int )); //host memory data
    int *dataB = (int *)malloc(M*N*sizeof(int )); //host memory data
    int *dataC =(int *)malloc(M*N*sizeof(int )); //host memory data

    int **dev_A ;  //device memory
    int **dev_B ;  //device memory
    int **dev_C ;  //device memory
    int *dev_dataA ;  //device memory  data
    int *dev_dataB ;  //device memory  data
    int *dev_dataC ;  //device memory  data

    cudaMalloc((void**)(&dev_A), M*sizeof(int*));
    cudaMalloc((void**)(&dev_dataA), M*N*sizeof(int));
    cudaMalloc((void**)(&dev_B), M*sizeof(int*));
    cudaMalloc((void**)(&dev_dataB), M*N*sizeof(int));
    cudaMalloc((void**)(&dev_C), M*sizeof(int*));
    cudaMalloc((void**)(&dev_dataC), M*N*sizeof(int));

    for(int i=0;i<M*N;i++)
    {
        dataA[i] = i;
        dataB[i] = i+1;
        dataC[i] =0;
    }

    cudaMemcpy((void*)(dev_dataA), (void*)(dataA), M*N*sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)(dev_dataB), (void*)(dataB), M*N*sizeof(int*), cudaMemcpyHostToDevice);


    for(int i=0;i<M;i++)
    {
        A[i] = dev_dataA + N*i;
        B[i] = dev_dataB + N*i;
        C[i] = dev_dataC + N*i;
    }


    cudaMemcpy((void*)(dev_A), (void*)(A), M*sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)(dev_B), (void*)(B), M*sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)(dev_C), (void*)(C), M*sizeof(int*), cudaMemcpyHostToDevice);

    dim3 threadPerBlock(16,16);
    dim3 numBlocks((N+threadPerBlock.x-1)/(threadPerBlock.x), (M+threadPerBlock.y-1)/(threadPerBlock.y));
    addMat<<<numBlocks,threadPerBlock>>>(dev_A,dev_B,dev_C);
    cudaMemcpy((void*)(dataC), (void*)(dev_dataC), M*N*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<M*N;i++)
        std::cout<<dataC[i]<<" ";
    cudaFree((void*)dev_dataC);
    cudaFree((void*)dev_C);
    free(C);
    free(dataC);
    cudaFree((void*)dev_dataB);
    cudaFree((void*)dev_B);
    free(B);
    free(dataB);
    cudaFree((void*)dev_dataA);
    cudaFree((void*)dev_A);
    free(A);
    free(dataA);
    getchar();
}
