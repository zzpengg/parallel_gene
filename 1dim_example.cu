#include<stdio.h>
#include<cuda.h>

//索引用到的緒構體
struct Index{
        int block, thread;
};

//核心:把索引寫入裝置記憶體
__global__ void prob_idx(Index id[]){
        int b=blockIdx.x;       //區塊索引
        int t=threadIdx.x;      //執行緒索引
        int n=blockDim.x;       //區塊中包含的執行緒數目
        int x=b*n+t;            //執行緒在陣列中對應的位置

        //每個執行緒寫入自己的區塊和執行緒索引.
        id[x].block=b;
        id[x].thread=t;
};

//主函式
int main(){
        Index* d;
        Index  h[3000];

        //配置裝置記憶體
        cudaMalloc((void**) &d, 3000*sizeof(Index));

        //呼叫裝置核心
        int g=1, b=1000, m=g*b;
        prob_idx<<<g,b>>>(d);

        //下載裝置記憶體內容到主機上
        cudaMemcpy(h, d, 3000*sizeof(Index), cudaMemcpyDeviceToHost);

        //顯示內容
        for(int i=0; i<m; i++){
            printf("h[%d]={block:%d, thread:%d}\n", i,h[i].block,h[i].thread);
        }

        //釋放裝置記憶體
        cudaFree(d);
        return 0;
}