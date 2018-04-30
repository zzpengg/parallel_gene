#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

#include <unistd.h>

#include <curand.h>
#include <curand_kernel.h>

#define ISLAND 6
#define POPULATION 20
#define FACILITY 20
#define GENERATION 1
#define CROSSOVER 0.6
#define MUTATION 0.03
#define MIGRATION 15
#define INDIVIDUAL 5

#define H 15 // BAY height
#define W 10 // BAY width

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

__global__ void randomData(curandState_t* states, short* GA){

    short b=blockIdx.x;       //區塊索引 == ISLAND
    short t=threadIdx.x;      //執行緒索引 == POPULATION
    short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
    short x=b*n+t;

    for(int j=0;j<FACILITY;j++){ // setup
        GA[x*FACILITY + j] = j;
    }

    int i; // shuffle
    for(i = 0; i < FACILITY; i++) {
        short k = curand(&states[x]) % FACILITY;
        int tmp = GA[x*FACILITY + i];
        GA[x*FACILITY + i] = GA[x*FACILITY + k];
        GA[x*FACILITY + k] = tmp;
    }
    
}

__global__ void randomBay(curandState_t* states, bool* GB){

    short b=blockIdx.x;       //區塊索引 == ISLAND
    short t=threadIdx.x;      //執行緒索引 == POPULATION
    short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
    short x=b*n+t;

    int i; // shuffle
    for(i = 0; i < FACILITY-1; i++) {
        GB[x*(FACILITY-1) + i] = curand(&states[x]) % 2;
    }
    
}

__global__ void calPosition(short *data, bool *bay, float *position){

    short b=blockIdx.x;       //區塊索引 == ISLAND
    short t=threadIdx.x;      //執行緒索引 == POPULATION
    short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
    short x=b*n+t;

    short posit = x * FACILITY;
    short bayposit = x * (FACILITY-1);
    // int posit=b*POPULATION*FACILITY+t*FACILITY;            //執行緒在陣列中對應的位置
    // int posofposit = b*POPULATION*FACILITY*2+t*FACILITY*2;

    for(int i=0;i<ISLAND*POPULATION*FACILITY*2;i++){
        position[i] = 0;
    }


    short len = 1;
    short next = 0;
    for(short f=0;f<FACILITY;f++){
        if(bay[bayposit+f] == 0){
            len = len + 1;
        }
        if(bay[bayposit+f] == 1 || f == FACILITY - 1 ){
            if(f == FACILITY - 1 && bay[bayposit+f] == 0){
                len = len - 1;
            }
            float x = W / 2.0 + next;

            for(short j=0;j<len;j++){

                position[posit*2+(f+j-len+1)*2] = x;

                float y = H / (len * 2.0) * ( (j * 2) + 1) ;

                position[posit*2+(f+j-len+1)*2+1] = y;
            }
            len = 1;

            next = next + W;
        }
    }
}

int main(){
    float START, END;
    START = clock();

    curandState_t* states;
    cudaMalloc((void**) &states, ISLAND * POPULATION * sizeof(curandState_t));
    // init seed
    init<<<ISLAND, POPULATION>>>(time(NULL), states);

    // generate random data
    short *GA;
    cudaMalloc((void**)&GA, ISLAND*POPULATION*FACILITY*sizeof(short));
    bool *GB;
    cudaMalloc((void**)&GB, ISLAND*POPULATION*(FACILITY-1)*sizeof(bool));

    randomData<<<ISLAND, POPULATION>>>(states, GA);
    randomBay<<<ISLAND, POPULATION>>>(states, GB);

    short data[ISLAND][POPULATION][FACILITY];
    bool bay[ISLAND][POPULATION][FACILITY-1];
    cudaMemcpy(data, GA, ISLAND*POPULATION*FACILITY*sizeof(short), cudaMemcpyDeviceToHost);
    cudaMemcpy(bay, GB, ISLAND*POPULATION*(FACILITY-1)*sizeof(bool), cudaMemcpyDeviceToHost);

    // print data
    for(int i=0;i<ISLAND;i++){
        for(int j=0;j<POPULATION;j++){
            for(int k=0;k<FACILITY;k++){
                printf("%hu ", data[i][j][k]);
            }
            printf("\n");
        }
    }

    // print bay
    for(int i=0;i<ISLAND;i++){
        for(int j=0;j<POPULATION;j++){
            for(int k=0;k<FACILITY-1;k++){
                printf("%d ", bay[i][j][k]);
            }
            printf("\n");
        }
    }

    FILE *fPtr;

    int ttt = FACILITY * (FACILITY-1) ;

	fPtr=fopen("cost.txt","r");
	int cost[FACILITY][FACILITY] = {0};
	int temp[ttt][3]; // cost
	for(int i=0;i<ttt;i++){
		fscanf(fPtr , "%d %d %d" , &temp[i][0], &temp[i][1], &temp[i][2]);
	}
	fclose(fPtr);
	for(int i=0;i<ttt;i++){ // 2 dimention cost
		cost[ temp[i][0]-1 ][ temp[i][1]-1] = temp[i][2];
	}

    for(int i=0;i<FACILITY;i++){
        for(int j=0;j<FACILITY;j++){
            printf("%d ", cost[i][j]);
        }
        printf("\n");
    }

    int *Gcost;
    cudaMalloc((void**)&Gcost, FACILITY*FACILITY*sizeof(int));
    cudaMemcpy(Gcost, cost, FACILITY*FACILITY*sizeof(int), cudaMemcpyHostToDevice);

    for(int gggggg=0;gggggg<GENERATION;gggggg++){ // generation start
        float *Gposition; 
        cudaMalloc((void**)&Gposition, ISLAND*POPULATION*FACILITY*2*sizeof(float));
        // calculate position
        calPosition<<<ISLAND, POPULATION>>>(GA, GB, Gposition);

        float position[ISLAND][POPULATION][FACILITY][2];
        cudaMemcpy(position, Gposition, ISLAND*POPULATION*FACILITY*2*sizeof(float), cudaMemcpyDeviceToHost);
        for(int i=0;i<ISLAND;i++){
            for(int p=0;p<POPULATION;p++){
                for(int f=0;f<FACILITY;f++){
                    for(int t=0;t<2;t++){
                        printf("%.2f ", position[i][p][f][t]);
                    }
                    printf("\n");
                }
            }
        }
    
    } // generation end


    END = clock();

    printf("%f\n", (END - START) / CLOCKS_PER_SEC);

    return 0;
}