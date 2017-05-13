#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <math.h>


#define ISLAND 1
#define POPULATION 10
#define FACILITY 6
#define GENERATION 30
#define CROSSOVER 0.6
#define MUTATION 0.03
#define MIGRATION 5
#define INDIVIDUAL 5

#define H 3 // BAY height
#define W 2 // BAY width

void shuffle(int* facility);

__global__ void calPosition(int *data, short int *bay, float *position){

  int b=blockIdx.x;       //區塊索引 == ISLAND
  int t=threadIdx.x;      //執行緒索引 == POPULATION
  int n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  int x=b*n+t;

  int posit = x * FACILITY;
  int bayposit = x * (FACILITY-1);
  // int posit=b*POPULATION*FACILITY+t*FACILITY;            //執行緒在陣列中對應的位置
  // int posofposit = b*POPULATION*FACILITY*2+t*FACILITY*2;

  for(int i=0;i<ISLAND*POPULATION*FACILITY*2;i++){
    position[i] = 0;
  }


			int len = 1;
			int next = 0;
			for(int f=0;f<FACILITY;f++){
				if(bay[bayposit+f] == 0){
					len = len + 1;
				}
				if(bay[bayposit+f] == 1 || f == FACILITY - 1 ){
					if(f == FACILITY - 1 && bay[bayposit+f] == 0){
						len = len - 1;
					}
					float x = W / 2.0 + next;

					for(int j=0;j<len;j++){

						position[posit*2+(f+j-len+1)*2] = x;

						float y = H / (len * 2.0) * ( (j * 2) + 1) ;

						position[posit*2+(f+j-len+1)*2+1] = y;
					}
					len = 1;

					next = next + W;
				}
			}
}

__global__ void calDistance(int *data, float *position, float *distance){

  int b=blockIdx.x;       //區塊索引 == ISLAND
  int t=threadIdx.x;      //執行緒索引 == POPULATION
  int n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  int x=b*n+t;

  int posit = x * FACILITY;

  // int posit=b*POPULATION*FACILITY+t*FACILITY;            //執行緒在陣列中對應的位置
  // int posofposit = b*POPULATION*FACILITY*2+t*FACILITY*2;

  for(int i=0;i<ISLAND*POPULATION*FACILITY*FACILITY;i++){
    distance[i] = 0;
  }


  for(int f=0;f<FACILITY;f++){
    // printf("\ndistance calculate facility%d\n", f);
    for(int j=f+1;j<FACILITY;j++){

      float x1 = position[ (posit + f)*2 ];
      float y1 = position[ (posit + f)*2 + 1];

      int x = data[ posit + f ];
      // printf("x = %d\n", x);
      float x2 = position[ (posit + j)*2 ];
      float y2 = position[ (posit + j)*2 + 1];
      int y = data[ posit + j ];
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

  int b=blockIdx.x;       //區塊索引 == ISLAND
  int t=threadIdx.x;      //執行緒索引 == POPULATION
  int n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  int x=b*n+t;

  int posit = x * FACILITY;

  // int posit=b*POPULATION*FACILITY+t*FACILITY;            //執行緒在陣列中對應的位置
  // int posofposit = b*POPULATION*FACILITY*2+t*FACILITY*2;

  // for(int i=0;i<ISLAND*POPULATION*FACILITY*FACILITY;i++){
  //   totalCost[i] = 0;
  // }


	for(int f=0;f<FACILITY;f++){
		for(int j=0;j<FACILITY;j++){
			totalCost[ (posit + f)*FACILITY + j] = cost[f*FACILITY + j] * distance[ (posit + f)*FACILITY + j];
		}
	}

}

__global__ void calOF(float *sumCost, float *minCost, float *totalCost){

  int b=blockIdx.x;       //區塊索引 == ISLAND
  int t=threadIdx.x;      //執行緒索引 == POPULATION
  int n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  int x=b*n+t;

  int posit = x * FACILITY;

  // calculate OF

	sumCost[x] = 0.0;
	minCost[x/POPULATION * 2] = 0.0;


			for(int f=0;f<FACILITY;f++){
				for(int j=0;j<FACILITY;j++){
					sumCost[x] += totalCost[ (posit + f)*FACILITY + j];
				}
			}
			if(x % POPULATION==0){
				minCost[(x/POPULATION)*2] = sumCost[x*FACILITY + 0];
				minCost[(x/POPULATION)*2 + 1] = 0;
			}else if(minCost[x/POPULATION] > sumCost[x]){
				minCost[(x/POPULATION)*2] = sumCost[x];
				minCost[(x/POPULATION)*2 + 1] = x % POPULATION;
			}


}


__global__ void calProbability(float *probability, float *totalPro, float *sumCost){

  int b=blockIdx.x;       //區塊索引 == ISLAND
  int t=threadIdx.x;      //執行緒索引 == POPULATION
  int n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  int x=b*n+t;

  int posit = x;

  probability[x] = (1.0 / sumCost[x]) / (totalPro[ x / POPULATION ]) ;

}


__global__ void crossOver(float *probability2, int *data, short int *bay, int *data2, short int *bay2, int *tem, int *tem2, int *Gyes, int *Gsss, int *Gcount, int *GGgetP, int *GGgetP2, float *test){

  int b=blockIdx.x;       //區塊索引 == ISLAND
  int t=threadIdx.x;      //執行緒索引 == POPULATION
  int n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  int x=b*n+t;

  int posit = x * 2 * FACILITY;
  int posit2 = (2*x+1) * FACILITY;
  int bayposit = x * 2 * (FACILITY-1);
  int bayposit2 = (2*x+1) * (FACILITY-1);

			float get = (tem[x] % 100) * 0.01;
      test[x] = 0.0;
      tem[x] = tem[x] % 100;
			int getP = 0;
			float get2 = tem2[x] % 100 * 0.01;
      tem2[x] = tem2[x] % 100;
			int getP2 = 0;
      GGgetP2[x] = -1;
			for(int p=0;p<POPULATION-1;p++){
				if(get >= probability2[ (x/POPULATION)*POPULATION + p ] && get < probability2[ (x/POPULATION)*POPULATION + p+1 ]){
					getP = p+1;
          GGgetP2[x] = (x/POPULATION)*POPULATION + p;
					break;
				}
				else if(p==POPULATION-2){
					getP = p+1;
					break;
				}
			}
      test[x] = probability2[ (x/POPULATION)*POPULATION + 1];

			for(int p=0;p<POPULATION-1;p++){
				if(get2 >= probability2[x/POPULATION*POPULATION + p] && get2 < probability2[x/POPULATION*POPULATION + p+1]){
					getP2 = p+1;

					break;
				}
				else if(p==POPULATION-2){
					getP2 = p+1;

					break;
				}
			}

			for(int f=0;f<FACILITY;f++){
				data2[ posit + f] = data[ x/POPULATION*POPULATION*FACILITY + getP*FACILITY + f];
				bay2[ (2 * x)*(FACILITY-1) + f] = bay[ x/POPULATION*POPULATION*(FACILITY-1) + getP*(FACILITY-1) + f];
			}




			for(int f=0;f<FACILITY;f++){
				data2[ posit2 + f ] = data[x/POPULATION*POPULATION*FACILITY + getP2*FACILITY + f];
				bay2[ (2 * x + 1)*(FACILITY-1) + f] = bay[x/POPULATION*POPULATION*(FACILITY-1) + getP2*(FACILITY-1) + f];
			}


      int tt = Gyes[x] % 100;
			float yes = tt * 0.01;
      Gyes[x] = tt;

			if(yes <= CROSSOVER){

				int sss = FACILITY - 1;
        int seq = Gsss[x] % sss;
        Gsss[x] = seq;


				int cross[4][2];

				cross[0][0] = data2[ posit + seq];
				cross[0][1] = data2[ posit2 + seq];
				cross[1][0] = data2[ posit + seq];
				cross[1][1] = data2[ posit2 + seq+1];
				cross[2][0] = data2[ posit + seq+1];
				cross[2][1] = data2[ posit2 + seq];
				cross[3][0] = data2[ posit+ seq+1];
				cross[3][1] = data2[ posit2 + seq+1];



				int temp = data2[ posit2 + seq];

				int temp2 = data2[posit2 + seq+1];

				data2[ posit2 + seq] = data2[ posit + seq];

				data2[ posit2 + seq+1] = data2[posit + seq+1];

				data2[posit + seq] = temp;
				data2[posit + seq+1] = temp2;



				int count = 0;
				for(int c=0;c<4;c++){
					if(cross[c][0] == cross[c][1]){
						count++;
					}
				}
        Gcount[x] = count;

				switch (count) {
					case 0:
						for(int c=0;c<FACILITY;c++){
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

						for(int c=0;c<FACILITY;c++){
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
						for(int c=0;c<4;c++){
							if(cross[c][0] == cross[c][1]){
								temp = cross[c][0];
							}
						}

						for(int c=0;c<4;c++){
							if(cross[c][0] != temp && cross[c][1] != temp){
								for(int f=0;f<FACILITY;f++){
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

						for(int c=0;c<4;c++){
							if(cross[c][0] != temp && cross[c][1] != temp){
								for(int f=0;f<FACILITY;f++){
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

  printf("data\n");
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY;f++){
				printf("%d ", data[i*POPULATION*FACILITY+p*FACILITY+f]);
			}
			printf("\n");
		}
		printf("\n");
	}

  printf("bay\n");
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY-1;f++){
				printf("%d ", bay[i*POPULATION*(FACILITY-1)+p*(FACILITY-1)+f]);
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
	FILE *fPtr;

	fPtr=fopen("cost.txt","r");
	int cost[FACILITY*FACILITY] = {0};
	int temp[15*3]; // cost
	for(int i=0;i<15;i++){
		for(int a=0;a<3;a++){
			fscanf(fPtr , "%d " , &temp[i*3 + a]);
		}
	}
	fclose(fPtr);
	for(int i=0;i<15;i++){ // 2 dimention cost
		cost[ (temp[i*3]-1)*FACILITY + temp[i*3+1]-1] = temp[ i*3 + 2];
	}
  printf("cost: \n");
  for(int i=0;i<FACILITY;i++){ // 2 dimention cost
    for(int j=0;j<FACILITY;j++){
      printf("%d ", cost[i*FACILITY + j]);
    }
    printf("\n");
	}
  int *Gcost;
  cudaMalloc((void**)&Gcost, FACILITY*FACILITY*sizeof(int));
  cudaMemcpy(Gcost, cost, FACILITY*FACILITY*sizeof(int), cudaMemcpyHostToDevice);


  float *Gposition;
  cudaMalloc((void**)&Gposition, ISLAND*POPULATION*FACILITY*2*sizeof(float));

  // int *Gposition2;
  // cudaMalloc((void**)&Gposition2, ISLAND*POPULATION*FACILITY*2*sizeof(int));

  int g=ISLAND, b=POPULATION;
  // int m=g*b;
  calPosition<<<g, b>>>(GA, GB, Gposition);

  float position[ISLAND*POPULATION*FACILITY*2];
  // int position2[ISLAND*POPULATION*FACILITY*2];

  // int data2[ISLAND*POPULATION*FACILITY];
  // short int bay2[ISLAND*POPULATION*(FACILITY-1)]; //bay
  //
  // cudaMemcpy(data2, GA, ISLAND*POPULATION*FACILITY*sizeof(int), cudaMemcpyDeviceToHost);
  //
  // printf("data2\n");
	// for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
	// 		for(int f=0;f<FACILITY;f++){
	// 			printf("%d ", data[i*POPULATION*FACILITY+p*FACILITY+f]);
	// 		}
	// 		printf("\n");
	// 	}
	// 	printf("\n");
	// }
  // cudaMemcpy(bay2, GB, ISLAND*POPULATION*(FACILITY-1)*sizeof(short int), cudaMemcpyDeviceToHost);
  // printf("bay2\n");
	// for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
	// 		for(int f=0;f<FACILITY-1;f++){
	// 			printf("%d ", bay2[i*POPULATION*FACILITY+p*(FACILITY-1)+f]);
	// 		}
	// 		printf("\n");
	// 	}
	// 	printf("\n");
	// }

  cudaMemcpy(position, Gposition, ISLAND*POPULATION*FACILITY*2*sizeof(float), cudaMemcpyDeviceToHost);

  // print position
	for(int i=0;i<ISLAND;i++){
		printf("island%d \n", i);
		for(int p=0;p<POPULATION;p++){
			printf("po%d = \n",p);
			for(int f=0;f<FACILITY;f++){
				for(int k=0;k<2;k++){
					printf("%f ", position[i*POPULATION*FACILITY*2+p*FACILITY*2+f*2+k]);
				}
				printf("\n");
			}
		}
	}

  for(int i=0;i<ISLAND*POPULATION*FACILITY*2;i++){
    printf("%f ", position[i]);
  }
  printf("\n");

  float distance[ISLAND*POPULATION*FACILITY*FACILITY] = {0};

  float *Gdistance;
  cudaMalloc((void**)&Gdistance, ISLAND*POPULATION*FACILITY*FACILITY*sizeof(float));


  calDistance<<<g, b>>>(GA, Gposition, Gdistance);

	cudaMemcpy(distance, Gdistance, ISLAND*POPULATION*FACILITY*FACILITY*sizeof(float), cudaMemcpyDeviceToHost);

  printf("\ncalculate distance end\n");

  // print distance
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
      printf("po%d: \n", p);
			for(int f=0;f<FACILITY;f++){
				for(int j=0;j<FACILITY;j++){
					printf("%f ", distance[ i*POPULATION*FACILITY*FACILITY + p*FACILITY*FACILITY + f*FACILITY + j ]);
				}
				printf("\n");
			}
		}
	}


  float totalCost[ISLAND][POPULATION][FACILITY][FACILITY] = {0.0};

  float *GtotalCost;
  cudaMalloc((void**)&GtotalCost, ISLAND*POPULATION*FACILITY*FACILITY*sizeof(float));

  calTotalcost<<<g, b>>>(Gdistance, Gcost, GtotalCost);

  cudaMemcpy(totalCost, GtotalCost, ISLAND*POPULATION*FACILITY*FACILITY*sizeof(float), cudaMemcpyDeviceToHost);

  // print totalCost
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
      printf("po%d: \n", p);
			for(int f=0;f<FACILITY;f++){
				for(int j=0;j<FACILITY;j++){
					printf("%f ", totalCost[i][p][f][j]);
				}
				printf("\n");
			}
		}
	}

  float *GsumCost;
  float sumCost[ISLAND][POPULATION]={0.0};

  cudaMalloc((void**)&GsumCost, ISLAND*POPULATION*sizeof(float));

  float *GminCost;
  float minCost[ISLAND][2];
  cudaMalloc((void**)&GminCost, ISLAND*2*sizeof(float));

  calOF<<<g, b>>>(GsumCost, GminCost, GtotalCost);

  cudaMemcpy(sumCost, GsumCost, ISLAND*POPULATION*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(minCost, GminCost, ISLAND*2*sizeof(float), cudaMemcpyDeviceToHost);

  printf("\n");
	for(int i=0;i<ISLAND;i++){
		printf("第%d島嶼: \n", i);
		for(int p=0;p<POPULATION;p++){
			printf("%d: ", p);
		  printf("sum = %f", sumCost[i][p]);
			printf("\n");
		}
	}


  int data2[ISLAND][POPULATION][FACILITY]; // facility
  int *Gdata2;
  cudaMalloc((void**)&Gdata2, ISLAND*POPULATION*FACILITY*sizeof(int));
	short int bay2[ISLAND][POPULATION][FACILITY]; //bay
  short int *Gbay2;
  cudaMalloc((void**)&Gbay2, ISLAND*POPULATION*FACILITY*sizeof(short int));

	float probability[ISLAND][POPULATION] = {0.0}; // �U�Ӿ��v
  float *Gprobability;
  cudaMalloc((void**)&Gprobability, ISLAND*POPULATION*sizeof(float));

	float totalPro[ISLAND] = {0.0};                // �`(�����˼�)
  float *GtotalPro;
  cudaMalloc((void**)&GtotalPro, ISLAND*sizeof(float));

  for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			totalPro[i] = totalPro[i] + (1.0 / sumCost[i][p]);
			printf("%f %f\n", totalPro[i], (1.0 / sumCost[i][p]));
		}
	}

  cudaMemcpy(GtotalPro, totalPro, ISLAND*sizeof(float), cudaMemcpyHostToDevice);


	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			printf("%f %f\n", totalPro[i], (1.0 / sumCost[i][p]));
		}
	}

  calProbability<<<ISLAND, POPULATION>>>(Gprobability, GtotalPro, GsumCost);

  cudaMemcpy(probability, Gprobability, ISLAND*POPULATION*sizeof(float), cudaMemcpyDeviceToHost);

	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			printf("%f %f %f \n", probability[i][p], (1.0 / sumCost[i][p]), totalPro[i]);
		}
	}


  float probability2[ISLAND][POPULATION] = {0.0};
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			for(int j=0;j<=p;j++){
				probability2[i][p] += probability[i][j];
			}
		}
	}

  float *Gprobability2;
  cudaMalloc((void**)&Gprobability2, ISLAND*POPULATION*sizeof(float));
  cudaMemcpy(Gprobability2, probability2, ISLAND*POPULATION*sizeof(float), cudaMemcpyHostToDevice);

	// print probability2 (Roulette)
	printf("probability2\n");
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			printf("%f ", probability2[i][p]);
		}
	}



  int *Gtem, *Gtem2, *Gyes, *Gsss;// choose two to crossover and if yes or not and choose area
  int tem[20], tem2[20], yes[20], sss[20];
  cudaMalloc((void**)&Gtem, 20*sizeof(int));
  cudaMalloc((void**)&Gtem2, 20*sizeof(int));
  cudaMalloc((void**)&Gyes, 20*sizeof(int));
  cudaMalloc((void**)&Gsss, 20*sizeof(int));

  for(int i=0;i<20;i++){
    tem[i] = rand();
    tem2[i] = rand();
    yes[i] = rand();
    sss[i] = rand();
    printf("%d %d %d %d\n", tem[i], tem2[i], yes[i], sss[i]);
  }

  cudaMemcpy(Gtem, tem, 20*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(Gtem2, tem2, 20*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(Gyes, yes, 20*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(Gsss, sss, 20*sizeof(int), cudaMemcpyHostToDevice);

  int *Gcount;
  cudaMalloc((void**)&Gcount, 10*sizeof(int));
  int *GetP, *GetP2;
  cudaMalloc((void**)&GetP, 10*sizeof(int));
  cudaMalloc((void**)&GetP2, 10*sizeof(int));
  int getP[10], getP2[10];
  float *Gtest;
  cudaMalloc((void**)&Gtest, 10*sizeof(float));
  float test[10] = {0.0};
  crossOver<<<ISLAND, POPULATION / 2>>>(Gprobability2, GA, GB, Gdata2, Gbay2, Gtem, Gtem2, Gyes, Gsss, Gcount, GetP, GetP2, Gtest);
  cudaMemcpy(data2, Gdata2, ISLAND*POPULATION*FACILITY*sizeof(int), cudaMemcpyDeviceToHost);

  int count[10] = {0};
  cudaMemcpy(tem, Gtem, 20*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(tem2, Gtem2, 20*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(yes, Gyes, 20*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(sss, Gsss, 20*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(count, Gcount, 10*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(getP, GetP, 10*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(getP2, GetP2, 10*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(test, Gtest, 10*sizeof(int), cudaMemcpyDeviceToHost);

  printf("count: \n");
  for(int i=0;i<10;i++){
    printf("%d ", count[i]);
  }

  printf("\nget: \n");
  for(int i=0;i<10;i++){
    printf("%d %d\n", getP[i], getP2[i]);
  }

  printf("\ntest: \n");
  for(int i=0;i<10;i++){
    printf("%f\n", test[i]);
  }

  printf("\nTEM: \n");
  for(int i=0;i<20;i++){
    printf("%d %d %d %d\n", tem[i], tem2[i], yes[i], sss[i]);
  }


  for(int i=0;i<ISLAND;i++){
    for(int p=0;p<POPULATION;p++){
      printf("\n交配結果(data2)%d\n", p);
      for(int f=0;f<FACILITY;f++){
        printf("%d ", data2[i][p][f]);
      }
      printf("\n");
    }
  }





  cudaFree(Gtem);
  cudaFree(Gtem2);
  cudaFree(GetP);
  cudaFree(GetP2);
  cudaFree(Gtest);
  cudaFree(Gdistance);
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
