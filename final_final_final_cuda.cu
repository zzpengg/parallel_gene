#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

#include <unistd.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>


#define ISLAND 10
#define POPULATION 50
#define FACILITY 20
#define GENERATION 8
#define CROSSOVER 0.6
#define MUTATION 0.03
#define MIGRATION 15
#define INDIVIDUAL 5

#define H 15 // BAY height
#define W 10 // BAY width

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

__global__ void calDistance(short *data, float *position, float *distance){

  short b=blockIdx.x;       //區塊索引 == ISLAND
  short t=threadIdx.x;      //執行緒索引 == POPULATION
  short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  short x=b*n+t;

  short posit = x * FACILITY;

  // int posit=b*POPULATION*FACILITY+t*FACILITY;            //執行緒在陣列中對應的位置
  // int posofposit = b*POPULATION*FACILITY*2+t*FACILITY*2;

  for(int i=0;i<ISLAND*POPULATION*FACILITY*FACILITY;i++){
    distance[i] = 0;
  }


  for(short f=0;f<FACILITY;f++){
    // printf("\ndistance calculate facility%d\n", f);
    for(short j=f+1;j<FACILITY;j++){

      float x1 = position[ (posit + f)*2 ];
      float y1 = position[ (posit + f)*2 + 1];

      short x = data[ posit + f ];
      // printf("x = %d\n", x);
      float x2 = position[ (posit + j)*2 ];
      float y2 = position[ (posit + j)*2 + 1];
      short y = data[ posit + j ];
      // printf("y= %d\n", y);
      if(y2 > y1){
        distance[ (posit + x)*FACILITY + y] = sqrt( (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ) ;
        distance[ (posit + y)*FACILITY + x] = sqrt( (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ) ;
      }
      else{
        distance[ (posit + x)*FACILITY + y] = sqrt( (x2 - x1) * (x2 - x1) + (y1 - y2) * (y1 - y2) ) ;
        distance[ (posit + y)*FACILITY + x] = sqrt( (x2 - x1) * (x2 - x1) + (y1 - y2) * (y1 - y2) ) ;
      }
    }
  }

}

__global__ void calTotalcost(float *distance, int *cost, float *totalCost){

  short b=blockIdx.x;       //區塊索引 == ISLAND
  short t=threadIdx.x;      //執行緒索引 == POPULATION
  short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  short x=b*n+t;

  short posit = x * FACILITY;

  // short posit=b*POPULATION*FACILITY+t*FACILITY;            //執行緒在陣列中對應的位置
  // int posofposit = b*POPULATION*FACILITY*2+t*FACILITY*2;

  // for(int i=0;i<ISLAND*POPULATION*FACILITY*FACILITY;i++){
  //   totalCost[i] = 0;
  // }


	for(short f=0;f<FACILITY;f++){
		for(short j=0;j<FACILITY;j++){
			totalCost[ (posit + f)*FACILITY + j] = cost[f*FACILITY + j] * distance[ (posit + f)*FACILITY + j];
		}
	}

}

__global__ void calOF(float *sumCost, float *totalCost){

    short b=blockIdx.x;       //區塊索引 == ISLAND
    short t=threadIdx.x;      //執行緒索引 == POPULATION
    short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
    short x=b*n+t;

    short posit = x * FACILITY;

    // calculate OF

	sumCost[x] = 0.0;
	// minCost[x/POPULATION * 2] = 0.0;


    for(short f=0;f<FACILITY;f++){
        for(short j=0;j<FACILITY;j++){
            sumCost[x] += totalCost[ (posit + f)*FACILITY + j];
        }
    }
    // if(x % POPULATION==0){
    //     minCost[(x/POPULATION)*2] = sumCost[x*FACILITY + 0];
    //     minCost[(x/POPULATION)*2 + 1] = 0;
    // }else if(minCost[x/POPULATION] > sumCost[x]){
    //     minCost[(x/POPULATION)*2] = sumCost[x];
    //     minCost[(x/POPULATION)*2 + 1] = x % POPULATION;
    // }


}

__global__ void calTotalPro(float *sumCost, float *totalPro){

    short b=blockIdx.x;       //區塊索引 == ISLAND
    short t=threadIdx.x;      //執行緒索引 == POPULATION
    short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
    short x=b*n+t;
    
    for(short p=0;p<POPULATION;p++){
        totalPro[x] = totalPro[x] + (1.0 / sumCost[x*ISLAND +  p]);
    }
}


__global__ void calProbability(float *probability, float *totalPro, float *sumCost){

  short b=blockIdx.x;       //區塊索引 == ISLAND
  short t=threadIdx.x;      //執行緒索引 == POPULATION
  short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  short x=b*n+t;


  probability[x] = (1.0 / sumCost[x]) / (totalPro[ x / POPULATION ]) ;

}

__global__ void crossOver(curandState_t* states,float *probability2, short *data, bool *bay, short *data2, bool *bay2){

    short b=blockIdx.x;       //區塊索引 == ISLAND
    short t=threadIdx.x;      //執行緒索引 == POPULATION
    short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
    short x=b*n+t;

    short posit = x * 2 * FACILITY;
    short posit2 = (2*x+1) * FACILITY;
    short bayposit = x * 2 * (FACILITY-1);
    short bayposit2 = (2*x+1) * (FACILITY-1);

    float get = ( curand(&states[x]) % 10000) * 0.0001;
    short getP = 0;
    float get2 = curand(&states[x]) % 10000 * 0.0001;
    short getP2 = 0;
    for(short p=0;p<POPULATION-1;p++){
        if(get >= probability2[ (x/POPULATION)*POPULATION + p ] && get < probability2[ (x/POPULATION)*POPULATION + p+1 ]){
            getP = p+1;
            break;
        }
        else if(p==POPULATION-2){
            getP = p+1;
            break;
        }
    }

    for(short p=0;p<POPULATION-1;p++){
        if(get2 >= probability2[x/POPULATION*POPULATION + p] && get2 < probability2[x/POPULATION*POPULATION + p+1]){
            getP2 = p+1;
            break;
        }
        else if(p==POPULATION-2){
            getP2 = p+1;
            break;
        }
    }

    for(short f=0;f<FACILITY;f++){
        data2[ posit + f] = data[ x/POPULATION*POPULATION*FACILITY + getP*FACILITY + f];
        bay2[ (2 * x)*(FACILITY-1) + f] = bay[ x/POPULATION*POPULATION*(FACILITY-1) + getP*(FACILITY-1) + f];
    }

    for(short f=0;f<FACILITY;f++){
        data2[ posit2 + f ] = data[x/POPULATION*POPULATION*FACILITY + getP2*FACILITY + f];
        bay2[ (2 * x + 1)*(FACILITY-1) + f] = bay[x/POPULATION*POPULATION*(FACILITY-1) + getP2*(FACILITY-1) + f];
    }


    int tt = curand(&states[x]) % 10000;
    float yes = tt * 0.0001;

    if(yes <= CROSSOVER){

        short sss = FACILITY - 1;
        int seq = curand(&states[x]) % sss;

        int cross[4][2];

        cross[0][0] = data2[ posit + seq];
        cross[0][1] = data2[ posit2 + seq];
        cross[1][0] = data2[ posit + seq];
        cross[1][1] = data2[ posit2 + seq+1];
        cross[2][0] = data2[ posit + seq+1];
        cross[2][1] = data2[ posit2 + seq];
        cross[3][0] = data2[ posit+ seq+1];
        cross[3][1] = data2[ posit2 + seq+1];



        short temp = data2[ posit2 + seq];

        short temp2 = data2[posit2 + seq+1];

        data2[ posit2 + seq] = data2[ posit + seq];

        data2[ posit2 + seq+1] = data2[posit + seq+1];

        data2[posit + seq] = temp;
        data2[posit + seq+1] = temp2;



        short count = 0;
        for(short c=0;c<4;c++){
            if(cross[c][0] == cross[c][1]){
                count++;
            }
        }

        switch (count) {
            case 0:
                for(short c=0;c<FACILITY;c++){
                    if(c != seq){
                        if(data2[posit + c] == cross[0][1]){
                            data2[ posit + c] = cross[0][0];
                        }
                        if(data2[posit + c] == cross[3][1]){
                            data2[ posit + c] = cross[3][0];
                        }
                    }
                    else{
                        c++;
                    }
                }

                for(short c=0;c<FACILITY;c++){
                    if(c != seq){
                        if(data2[posit2 + c] == cross[0][0]){
                            data2[ posit2 + c] = cross[0][1];
                        }
                        if(data2[posit2 + c] == cross[3][0]){
                            data2[ posit2 + c] = cross[3][1];
                        }
                    }
                    else{
                        c++;
                    }
                }
                break;
            case 1:
                temp = 99;
                for(short c=0;c<4;c++){
                    if(cross[c][0] == cross[c][1]){
                        temp = cross[c][0];
                    }
                }

                for(short c=0;c<4;c++){
                    if(cross[c][0] != temp && cross[c][1] != temp){
                        for(short f=0;f<FACILITY;f++){
                            if(f != seq){
                                if(data2[posit + f] == cross[c][1]){
                                    data2[ posit + f] = cross[c][0];
                                }
                            }
                            else{
                                f++;
                            }
                        }
                    }
                }

                for(short c=0;c<4;c++){
                    if(cross[c][0] != temp && cross[c][1] != temp){
                        for(short f=0;f<FACILITY;f++){
                            if(f != seq){
                                if(data2[posit2 + f] == cross[c][0]){
                                    data2[ posit2 + f] = cross[c][1];
                                }
                            }
                            else{
                                f++;
                            }
                        }
                    }
                }
                break;
            case 2:
                break;
        }




        temp = bay2[bayposit2 + seq];
        temp2 = bay2[bayposit2 + seq+1];
        bay2[bayposit2 + seq]   = bay2[bayposit + seq];
        bay2[bayposit2 + seq+1] = bay2[bayposit + seq+1];
        bay2[bayposit + seq]   = bay2[bayposit2 + seq];
        bay2[bayposit + seq+1] = bay2[bayposit2 + seq+1];
    }else {

    }



}

__global__ void mutation(curandState_t *states, short *data2){
  int b=blockIdx.x;       //區塊索引 == ISLAND
  int t=threadIdx.x;      //執行緒索引 == POPULATION
  int n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  int x=b*n+t;

  int posit = x * FACILITY;

  float yes = (curand(&states[x]) % 10000) * 0.0001;
	// fprintf(FIN, "取得%f \n", yes);
	if(yes < MUTATION){
		// fprintf(FIN, "第%d突變\n", p);
		short get = curand(&states[x]) % FACILITY;
		short get2 = curand(&states[x]) % FACILITY;
		short temp = data2[posit + get];
		data2[posit + get] = data2[posit + get2];
		data2[posit + get2] = temp;
	}else {
	}
}

__global__ void mutationBay(curandState_t *states, bool *bay2){
  short b=blockIdx.x;       //區塊索引 == ISLAND
  short t=threadIdx.x;      //執行緒索引 == POPULATION
  short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  short x=b*n+t;

  short posit = x * (FACILITY - 1);

	float yes = curand(&states[x]) % 10000 * 0.0001 ;
	if(yes < MUTATION){
		short get = curand(&states[x]) % (FACILITY - 1);
		if(bay2[posit + get] == 0){
			bay2[posit + get] = 1;
		}else {
			bay2[posit + get] = 0;
		}
	}
}

// __global__ void migration(short *data2, bool *bay2, short *temp3, bool *temp4, int *indexCost,float *sumCost){
//     short b=blockIdx.x;       //區塊索引 == ISLAND
//     short t=threadIdx.x;      //執行緒索引 == POPULATION
//     short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
//     short x=b*n+t;

//     short posit = x * FACILITY ; // island
//     short positBay = x * (FACILITY - 1);

//     for(int f=0;f<FACILITY;f++){
//         temp3[posit +  f] = data2[posit + indexCost[x]*FACILITY + f] ;
//     }
//     for(int f=0;f<FACILITY-1;f++){
//         temp4[positBay + f] = bay2[positBay + indexCost[x]*(FACILITY-1) +  f];
//     }

//     __syncthreads();

//     if(posit == 0){
//         int backP = indexCost[(ISLAND-1)*POPULATION + x];
//         int frontP = indexCost[posit + x];
//         for(int f=0;f<FACILITY;f++){
//             data2[posit +  frontP*FACILITY + f] = temp3[(x/POPULATION -1)*POPULATION*FACILITY * + backP*FACILITY + f];
//         }
//         for(int f=0;f<FACILITY-1;f++){
//             bay2[posit + frontP*FACILITY + f] = temp4[(ISLAND-1)*POPULATION*FACILITY +  backP*FACILITY +  f];
//         }
//     }else {
//         int backP = indexCost[i-1 + x];
//         int frontP = indexCost[i + x];
//         for(int f=0;f<FACILITY;f++){
//             data2[i*POPULATION*FACILITY +  frontP*FACILITY +  f] = temp3[(ISLAND-1)*POPULATION*FACILITY +  backP*FACILITY +  f];
//         }
//         for(int f=0;f<FACILITY-1;f++){
//             bay2[i*POPULATION*FACILITY +  frontP*FACILITY +  f] = temp4[(ISLAND-1)*POPULATION*FACILITY +  backP*FACILITY +  f];
//         }
//     } // else end

// }

__global__ void parent_to_child(short *data, short *data2, bool *bay, bool *bay2){
    short b=blockIdx.x;       //區塊索引 == ISLAND
    short t=threadIdx.x;      //執行緒索引 == POPULATION
    short n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
    short x=b*n+t;

    short posit = x * FACILITY ;
    short positBay = x * (FACILITY - 1);

    for(int f=0;f<FACILITY;f++){
        data[posit + f] = data2[posit + f];
    }

    for(int f=0;f<FACILITY-1;f++){
        bay[positBay + f] = bay2[positBay + f];
    }

}





int main(){

    double START, END;
    START = clock();

    /* CUDA's random number library uses curandState_t to keep track of the seed value
        we will store a random state for every thread  */
    curandState_t* states;

    /* allocate space on the GPU for the random states */
    cudaMalloc((void**) &states, ISLAND * POPULATION * sizeof(curandState_t));

    /* invoke the GPU to initialize all of the random states */
    init<<<ISLAND, POPULATION>>>(time(NULL), states);

    // generate random data
    short *GA;
    cudaMalloc((void**)&GA, ISLAND*POPULATION*FACILITY*sizeof(short));
    bool *GB;
    cudaMalloc((void**)&GB, ISLAND*POPULATION*(FACILITY-1)*sizeof(bool));

    randomData<<<ISLAND, POPULATION>>>(states, GA);
    randomBay<<<ISLAND, POPULATION>>>(states, GB);

    short data[ISLAND][POPULATION][FACILITY];
    cudaMemcpy(data, GA, ISLAND*POPULATION*FACILITY*sizeof(short), cudaMemcpyDeviceToHost);
    bool bay[ISLAND][POPULATION][FACILITY-1];
    cudaMemcpy(bay, GB, ISLAND*POPULATION*(FACILITY-1)*sizeof(bool), cudaMemcpyDeviceToHost);

    // print data
    // for(int i=0;i<ISLAND;i++){
    //     for(int j=0;j<POPULATION;j++){
    //         for(int k=0;k<FACILITY;k++){
    //             printf("%hu ", data[i][j][k]);
    //         }
    //         printf("\n");
    //     }
    // }

    // print bay
    // for(int i=0;i<ISLAND;i++){
    //     for(int j=0;j<POPULATION;j++){
    //         for(int k=0;k<FACILITY-1;k++){
    //             printf("%d ", bay[i][j][k]);
    //         }
    //         printf("\n");
    //     }
    // }

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

    int *Gcost;
    cudaMalloc((void**)&Gcost, FACILITY*FACILITY*sizeof(int));
    cudaMemcpy(Gcost, cost, FACILITY*FACILITY*sizeof(int), cudaMemcpyHostToDevice);

    for(short gggggg=0;gggggg<GENERATION;gggggg++){ // generation    
        float *Gposition; 
        cudaMalloc((void**)&Gposition, ISLAND*POPULATION*FACILITY*2*sizeof(float));
        // calculate position
        calPosition<<<ISLAND, POPULATION>>>(GA, GB, Gposition);
        float position[ISLAND*POPULATION*FACILITY*2];
        cudaMemcpy(position, Gposition, ISLAND*POPULATION*FACILITY*2*sizeof(float), cudaMemcpyDeviceToHost);

        float distance[ISLAND*POPULATION*FACILITY*FACILITY] = {0};

        float *Gdistance;
        cudaMalloc((void**)&Gdistance, ISLAND*POPULATION*FACILITY*FACILITY*sizeof(float));

        // distance
        calDistance<<<ISLAND, POPULATION>>>(GA, Gposition, Gdistance);

        cudaMemcpy(distance, Gdistance, ISLAND*POPULATION*FACILITY*FACILITY*sizeof(float), cudaMemcpyDeviceToHost);

        float totalCost[ISLAND][POPULATION][FACILITY][FACILITY] = {0.0};

        float *GtotalCost;
        cudaMalloc((void**)&GtotalCost, ISLAND*POPULATION*FACILITY*FACILITY*sizeof(float));

        // totalcost
        calTotalcost<<<ISLAND, POPULATION>>>(Gdistance, Gcost, GtotalCost);

        cudaMemcpy(totalCost, GtotalCost, ISLAND*POPULATION*FACILITY*FACILITY*sizeof(float), cudaMemcpyDeviceToHost);
    
        float *GsumCost;
        float sumCost[ISLAND][POPULATION]={0.0};

        cudaMalloc((void**)&GsumCost, ISLAND*POPULATION*sizeof(float));

        // float *GminCost;
        // float minCost[ISLAND][2];
        // cudaMalloc((void**)&GminCost, ISLAND*2*sizeof(float));

        // of
        calOF<<<ISLAND, POPULATION>>>(GsumCost, GtotalCost);

        cudaMemcpy(sumCost, GsumCost, ISLAND*POPULATION*sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy(minCost, GminCost, ISLAND*2*sizeof(float), cudaMemcpyDeviceToHost);

        short data2[ISLAND][POPULATION][FACILITY]; // facility
        short *Gdata2;
        cudaMalloc((void**)&Gdata2, ISLAND*POPULATION*FACILITY*sizeof(short));
        bool bay2[ISLAND][POPULATION][FACILITY-1]; //bay
        bool *Gbay2;
        cudaMalloc((void**)&Gbay2, ISLAND*POPULATION*(FACILITY-1)*sizeof(bool));

        float probability[ISLAND][POPULATION] = {0.0};

        float *Gprobability;
        cudaMalloc((void**)&Gprobability, ISLAND*POPULATION*sizeof(float));

        // float totalPro[ISLAND] = {0.0};                
        float *GtotalPro;
        cudaMalloc((void**)&GtotalPro, ISLAND*sizeof(float));

        calTotalPro<<<1, ISLAND>>>(GsumCost, GtotalCost);

        // cudaMemcpy(GtotalPro, totalPro, ISLAND*sizeof(float), cudaMemcpyHostToDevice);

        calProbability<<<ISLAND, POPULATION>>>(Gprobability, GtotalPro, GsumCost);

        cudaMemcpy(probability, Gprobability, ISLAND*POPULATION*sizeof(float), cudaMemcpyDeviceToHost);
    
        float probability2[ISLAND][POPULATION] = {0.0};
        for(short i=0;i<ISLAND;i++){
            for(short p=0;p<POPULATION;p++){
                for(short j=0;j<=p;j++){
                    probability2[i][p] += probability[i][j];
                }
            }
        }

        float *Gprobability2;
        cudaMalloc((void**)&Gprobability2, ISLAND*POPULATION*sizeof(float));
        cudaMemcpy(Gprobability2, probability2, ISLAND*POPULATION*sizeof(float), cudaMemcpyHostToDevice);
    
        crossOver<<<ISLAND, POPULATION / 2>>>(states, Gprobability2, GA, GB, Gdata2, Gbay2);

        mutation<<<ISLAND, POPULATION>>>(states, Gdata2);



        // migration
	if( (gggggg+1) % MIGRATION == 0 && (gggggg+1) != 0 && ISLAND > 1){
		// printf("***migration***\n");

        // int temp3[ISLAND][POPULATION/2][FACILITY];
        // short temp4[ISLAND][POPULATION/2][FACILITY-1];
        // int indexCost[ISLAND][POPULATION];

        cudaMemcpy(data2, Gdata2, ISLAND*POPULATION*FACILITY*sizeof(short), cudaMemcpyDeviceToHost);
        cudaMemcpy(bay2, Gbay2, ISLAND*POPULATION*(FACILITY-1)*sizeof(bool), cudaMemcpyDeviceToHost);

        int temp3[ISLAND][POPULATION/2][FACILITY];
		short temp4[ISLAND][POPULATION/2][FACILITY-1];
		int indexCost[ISLAND][POPULATION];

		for(int i=0;i<ISLAND;i++){
			for(int p=0;p<POPULATION;p++){
				indexCost[i][p] = p;
			}
		}

		// bubble sort
		// float temp;
		for(int k=0;k<ISLAND;k++){
			for(int i=POPULATION-1; i>=1; i--){
	      for(int j=0; j<=i-1; j++){
	        if(sumCost[k][j] > sumCost[k][j+1]){
							int temp2 = indexCost[k][j];
	            indexCost[k][j] = indexCost[k][j+1];
	            indexCost[k][j+1] = temp2;
	        }
	      }
	    }
		}

		// print sorted index
		// for(int i=0;i<ISLAND;i++){
		// 	for(int p=0;p<POPULATION;p++){
		// 		printf("%d ", indexCost[i][p]);
		// 	}
		// 	printf("\n");
		// }

		int countP = 0;
		for(int i=0;i<ISLAND;i++){
			while(countP < INDIVIDUAL){
				for(int p=0;p<POPULATION;p++){
					if(p == indexCost[i][countP]){
						for(int f=0;f<FACILITY;f++){
							temp3[i][countP][f] = data2[i][p][f];
						}
						for(int f=0;f<FACILITY-1;f++){
							temp4[i][countP][f] = bay2[i][p][f];
						}
						countP++;
						break;
					}
				} // population end
			}
			countP = 0;
		} // island end

		for(int i=0;i<ISLAND;i++){
			if(i==0){
				for(int k=0;k<POPULATION/2;k++){
					int backP = indexCost[ISLAND-1][k];
					int frontP = indexCost[i][k];
					for(int f=0;f<FACILITY;f++){
						data2[i][frontP][f] = temp3[ISLAND-1][backP][f];
					}
					for(int f=0;f<FACILITY-1;f++){
						bay2[i][frontP][f] = temp4[ISLAND-1][backP][f];
					}
				}
			}else {
				for(int k=0;k<POPULATION/2;k++){
					int backP = indexCost[i-1][k];
					int frontP = indexCost[i][k];
					// int p = indexCost[i][k];
					for(int f=0;f<FACILITY;f++){
						data2[i][frontP][f] = temp3[ISLAND-1][backP][f];
					}
					for(int f=0;f<FACILITY-1;f++){
						bay2[i][frontP][f] = temp4[ISLAND-1][backP][f];
					}
				}
			} // else end

		} // for end

        // cudaMemcpy(GindexCost, indexCost, ISLAND*POPULATION*sizeof(int));

        // migration <<< ISLAND, POPULATION >>> (Gdata2, Gbay2, Gtemp3, Gtemp4, GindexCost, GsumCost);
		
	} // if migration end

        // if(gggggg==GENERATION-1){
        if(1==1){

            cudaMemcpy(data2, Gdata2, ISLAND*POPULATION*FACILITY*sizeof(short), cudaMemcpyDeviceToHost);
            cudaMemcpy(bay2, Gbay2, ISLAND*POPULATION*(FACILITY-1)*sizeof(bool), cudaMemcpyDeviceToHost);
            int answerPos[2];
            float answer;
            answerPos[0] = 0;
            answerPos[1] = 0;
            answer = sumCost[0][0];
            for(int i=0;i<ISLAND;i++){
                // printf("第%d島嶼(OF): \n", i);
                for(int p=0;p<POPULATION;p++){
                    // printf("%f ", sumCost[i][p]);
                    if(sumCost[i][p] < answer && sumCost[i][p] != 0){
                        answerPos[0] = i;
                        answerPos[1] = p;
                        answer = sumCost[i][p];
                    }
                    // printf("\n");
                }
            }
            
            for(int i=0;i<FACILITY;i++){
                printf("%d ", data2[ answerPos[0] ][ answerPos[1] ][i]);
            }
            printf("\n");
            for(int i=0;i<FACILITY-1;i++){
                printf("%d  ", bay2[ answerPos[0] ][ answerPos[1] ][i]);
            }
            printf("最小: %d %d = %f\n", answerPos[0], answerPos[1], answer);
        }

        // parent to child
        parent_to_child<<<ISLAND, POPULATION>>>(GA, Gdata2, GB, Gbay2);


        cudaFree(Gposition);
        cudaFree(Gdistance);
        cudaFree(GtotalCost);
        cudaFree(GsumCost);
        // cudaFree(GminCost);
        cudaFree(Gdata2);
        cudaFree(Gbay2);
        cudaFree(Gprobability);
        cudaFree(GtotalPro);
    
    }
    cudaFree(GA); // free GA gpu_data
    cudaFree(GB); // free GB gpu_bay
    cudaFree(Gcost);
    

    END = clock();
    printf("程式執行所花費： %lf S\n", (double)clock()/CLOCKS_PER_SEC);
    printf("進行運算所花費的時間： %lf S\n", (END - START) / CLOCKS_PER_SEC);

    return 0;

}
