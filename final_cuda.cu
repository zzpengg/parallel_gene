#include <stdio.h>
#include <cuda.h>
#include <time.h>


#define ISLAND 1
#define POPULATION 10
#define FACILITY 6

#define H 3 // BAY height
#define W 2 // BAY width

void shuffle(int* facility);
__global__ void calPosition(int *data, short int *bay, int *position){

  int b=blockIdx.x;       //區塊索引 == ISLAND
  int t=threadIdx.x;      //執行緒索引 == POPULATION
  // int n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  int posit=b*POPULATION*FACILITY+t*FACILITY;            //執行緒在陣列中對應的位置
  int posofposit = b*POPULATION*FACILITY*2+t*FACILITY*2;

  for(int i=0;i<ISLAND*POPULATION*FACILITY*2;i++){
    position[i] = 0;
  }


			int len = 1;
			int next = 0;
			for(int f=0;f<FACILITY;f++){
				if(bay[posit+f] == 0){
					len = len + 1;
				}
				if(bay[posit+f] == 1 || f == FACILITY - 1 ){
					if(f == FACILITY - 1 && bay[posit+f] == 0){
						len = len - 1;
					}
					float x = W / 2.0 + next;

					for(int j=0;j<len;j++){

						position[posofposit+(f+j-len+1)] = b;

						float y = H / (len * 2.0) * ( (j * 2) + 1) ;

						position[posofposit+(f+j-len+1)+1] = t;
					}
					len = 1;

					next = next + W;
				}
			}
}

int main(){
  srand(time(NULL));

  int data[ISLAND*POPULATION*FACILITY];
  short int bay[ISLAND*POPULATION*(FACILITY-1)]; //bay

  int facility[FACILITY];

  for(int i=0;i<ISLAND;i++){ // shuffle the sorted facility
		printf("new island%d\n", i);
		for(int p=0;p<POPULATION;p++){
			for(int t=0;t<FACILITY;t++){
		    facility[t] = t;
			}
			shuffle(facility);
			// for(int t=0;t<FACILITY;t++){
			// 	printf("%d ", facility[t]);
			// }
			for(int f=0;f<FACILITY;f++){
				data[i*POPULATION*FACILITY+p*FACILITY+f] = facility[f];
				printf("%d ", data[i*POPULATION*FACILITY+p*FACILITY+f]);
			}
			printf("\n");
			for(int b=0;b<FACILITY-1;b++){
				int j = rand() % 2;
		    bay[i*POPULATION*FACILITY+p*(FACILITY-1)+b] = j;
			}
		}
	}

  printf("新的子代\n");
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY;f++){
				printf("%d ", data[i*POPULATION*FACILITY+p*FACILITY+f]);
			}
			printf("\n");
		}
		printf("\n");
	}

  int *GA;
  short int *GB;
  cudaMalloc((void**)&GA, ISLAND*POPULATION*FACILITY*sizeof(int));
	cudaMemcpy(GA, data, ISLAND*POPULATION*FACILITY*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&GB, ISLAND*POPULATION*(FACILITY-1)*sizeof(short int));
	cudaMemcpy(GB, bay, ISLAND*POPULATION*(FACILITY-1)*sizeof(short int), cudaMemcpyHostToDevice);

  // read ther cost
	// FILE *fPtr;
  //
	// fPtr=fopen("cost.txt","r");
	// int cost[FACILITY][FACILITY] = {0};
	// int temp[15][3]; // cost
	// for(int i=0;i<15;i++){
	// 	for(int a=0;a<3;a++){
	// 		fscanf(fPtr , "%d " , &temp[i][a]);
	// 	}
	// }
	// fclose(fPtr);
	// for(int i=0;i<15;i++){ // 2 dimention cost
	// 	cost[temp[i][0]-1][temp[i][1]-1] = temp[i][2];
	// }

  float *Gposition;
  cudaMalloc((void**)&Gposition, ISLAND*POPULATION*FACILITY*2*sizeof(float));

  int *Gposition2;
  cudaMalloc((void**)&Gposition2, ISLAND*POPULATION*FACILITY*2*sizeof(int));

  int g=ISLAND, b=POPULATION;
  // int m=g*b;
  calPosition<<<g, b>>>(GA, GB, Gposition2);

  float position[ISLAND*POPULATION*FACILITY*2];
  int position2[ISLAND*POPULATION*FACILITY*2];

  int data2[ISLAND*POPULATION*FACILITY];
  short int bay2[ISLAND*POPULATION*(FACILITY-1)]; //bay

  cudaMemcpy(data2, GA, ISLAND*POPULATION*FACILITY*sizeof(int), cudaMemcpyDeviceToHost);

  printf("data2\n");
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY;f++){
				printf("%d ", data[i*POPULATION*FACILITY+p*FACILITY+f]);
			}
			printf("\n");
		}
		printf("\n");
	}
  cudaMemcpy(bay2, GB, ISLAND*POPULATION*(FACILITY-1)*sizeof(short int), cudaMemcpyDeviceToHost);
  printf("bay2\n");
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY-1;f++){
				printf("%d ", bay2[i*POPULATION*FACILITY+p*(FACILITY-1)+f]);
			}
			printf("\n");
		}
		printf("\n");
	}

  cudaMemcpy(position2, Gposition2, ISLAND*POPULATION*FACILITY*2*sizeof(int), cudaMemcpyDeviceToHost);

  // print position
	for(int i=0;i<ISLAND;i++){
		printf("island%d \n", i);
		for(int p=0;p<POPULATION;p++){
			printf("po%d = \n",p);
			for(int f=0;f<FACILITY;f++){
				for(int k=0;k<2;k++){
					printf("%d ", position2[i*POPULATION*FACILITY*2+p*FACILITY*2+f*2+k]);
				}
				printf("\n");
			}
		}
	}

  for(int i=0;i<ISLAND*POPULATION*FACILITY;i++){
    printf("%d ", position2[i]);
  }

  cudaFree(Gposition);
  return 0;

}

void shuffle(int* facility) { // ���ñƧǦn��facility
    int i;
    for(i = 0; i < FACILITY; i++) {
        int j = rand() % FACILITY;
        int tmp = facility[i];
        facility[i] = facility[j];
        facility[j] = tmp;
    }
}
