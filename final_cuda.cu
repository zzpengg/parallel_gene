#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <math.h>


#define ISLAND 10
#define POPULATION 50
#define FACILITY 20
#define GENERATION 10
#define CROSSOVER 0.6
#define MUTATION 0.03
#define MIGRATION 15
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

__global__ void mutation(int *data2, int *mutaYes, int *mutaTem, int *mutaTem2){
  int b=blockIdx.x;       //區塊索引 == ISLAND
  int t=threadIdx.x;      //執行緒索引 == POPULATION
  int n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  int x=b*n+t;

  int posit = x * FACILITY;

  float yes = (mutaYes[x] % 100) * 0.01;
	// fprintf(FIN, "取得%f \n", yes);
	if(yes < MUTATION){
		// fprintf(FIN, "第%d突變\n", p);
		int get = mutaTem[x] % FACILITY;
		int get2 = mutaTem2[x] % FACILITY;
		int temp = data2[posit + get];
		data2[posit + get] = data2[posit + get2];
		data2[posit + get2] = temp;
	}else {
	}
}

__global__ void mutationBay(short int *bay2, int *mutaBayYes, int *mutaBayTem){
  int b=blockIdx.x;       //區塊索引 == ISLAND
  int t=threadIdx.x;      //執行緒索引 == POPULATION
  int n=blockDim.x;       //區塊中包含的執行緒數目 == num of ISLAND
  int x=b*n+t;

  int posit = x * (FACILITY - 1);

	float yes = mutaBayYes[x] % 100 * 0.01 ;
	if(yes < MUTATION){
		int get = mutaBayTem[x] % (FACILITY - 1);
		if(bay2[posit + get] == 0){
			bay2[posit + get] = 1;
		}else {
			bay2[posit + get] = 0;
		}
	}
}

int main(){
  double START,END;
  START = clock();
  srand(time(NULL));

  int data[ISLAND][POPULATION][FACILITY];
  short int bay[ISLAND][POPULATION][FACILITY-1]; //bay

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
				data[i][p][f] = facility[f];
				printf("%d ", data[i][p][f]);
			}
			printf("\n");
			for(int b=0;b<FACILITY-1;b++){
				int j = rand() % 2;
		    bay[i][p][b] = j;
			}
		}
	}

  printf("data\n");
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY;f++){
				printf("%d ", data[i][p][f]);
			}
			printf("\n");
		}
		printf("\n");
	}

  printf("bay\n");
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY-1;f++){
				printf("%d ", bay[i][p][f]);
			}
			printf("\n");
		}
		printf("\n");
	}

  // int *GA;
  // short int *GB;
  // cudaMalloc((void**)&GA, ISLAND*POPULATION*FACILITY*sizeof(int));
	// cudaMemcpy(GA, data, ISLAND*POPULATION*FACILITY*sizeof(int), cudaMemcpyHostToDevice);
  //
	// cudaMalloc((void**)&GB, ISLAND*POPULATION*(FACILITY-1)*sizeof(short int));
	// cudaMemcpy(GB, bay, ISLAND*POPULATION*(FACILITY-1)*sizeof(short int), cudaMemcpyHostToDevice);

  // read ther cost
	FILE *fPtr;

  int ttt = FACILITY * (FACILITY-1) / 2;

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
  printf("cost: \n");
  for(int i=0;i<FACILITY;i++){ // 2 dimention cost
    for(int j=0;j<FACILITY;j++){
      printf("%d ", cost[i][j]);
    }
    printf("\n");
	}
  int *Gcost;
  cudaMalloc((void**)&Gcost, FACILITY*FACILITY*sizeof(int));
  cudaMemcpy(Gcost, cost, FACILITY*FACILITY*sizeof(int), cudaMemcpyHostToDevice);


  for(int gggggg=0;gggggg<GENERATION;gggggg++){ // generation

  printf("\n*****%d的generation*****\n", gggggg);
  int *GA;
  short int *GB;
  cudaMalloc((void**)&GA, ISLAND*POPULATION*FACILITY*sizeof(int));
	cudaMemcpy(GA, data, ISLAND*POPULATION*FACILITY*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&GB, ISLAND*POPULATION*(FACILITY-1)*sizeof(short int));
	cudaMemcpy(GB, bay, ISLAND*POPULATION*(FACILITY-1)*sizeof(short int), cudaMemcpyHostToDevice);


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
	// for(int i=0;i<ISLAND;i++){
	// 	printf("island%d \n", i);
	// 	for(int p=0;p<POPULATION;p++){
	// 		printf("po%d = \n",p);
	// 		for(int f=0;f<FACILITY;f++){
	// 			for(int k=0;k<2;k++){
	// 				printf("%f ", position[i*POPULATION*FACILITY*2+p*FACILITY*2+f*2+k]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	// }

  // for(int i=0;i<ISLAND*POPULATION*FACILITY*2;i++){
  //   printf("%f ", position[i]);
  // }
  printf("\n");

  float distance[ISLAND*POPULATION*FACILITY*FACILITY] = {0};

  float *Gdistance;
  cudaMalloc((void**)&Gdistance, ISLAND*POPULATION*FACILITY*FACILITY*sizeof(float));


  calDistance<<<g, b>>>(GA, Gposition, Gdistance);

	cudaMemcpy(distance, Gdistance, ISLAND*POPULATION*FACILITY*FACILITY*sizeof(float), cudaMemcpyDeviceToHost);

  printf("\ncalculate distance end\n");

  // print distance
	// for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
  //     printf("po%d: \n", p);
	// 		for(int f=0;f<FACILITY;f++){
	// 			for(int j=0;j<FACILITY;j++){
	// 				printf("%f ", distance[ i*POPULATION*FACILITY*FACILITY + p*FACILITY*FACILITY + f*FACILITY + j ]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	// }


  float totalCost[ISLAND][POPULATION][FACILITY][FACILITY] = {0.0};

  float *GtotalCost;
  cudaMalloc((void**)&GtotalCost, ISLAND*POPULATION*FACILITY*FACILITY*sizeof(float));

  calTotalcost<<<g, b>>>(Gdistance, Gcost, GtotalCost);

  cudaMemcpy(totalCost, GtotalCost, ISLAND*POPULATION*FACILITY*FACILITY*sizeof(float), cudaMemcpyDeviceToHost);

  // print totalCost
	// for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
  //     printf("po%d: \n", p);
	// 		for(int f=0;f<FACILITY;f++){
	// 			for(int j=0;j<FACILITY;j++){
	// 				printf("%f ", totalCost[i][p][f][j]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	// }

  float *GsumCost;
  float sumCost[ISLAND][POPULATION]={0.0};

  cudaMalloc((void**)&GsumCost, ISLAND*POPULATION*sizeof(float));

  float *GminCost;
  float minCost[ISLAND][2];
  cudaMalloc((void**)&GminCost, ISLAND*2*sizeof(float));

  calOF<<<g, b>>>(GsumCost, GminCost, GtotalCost);

  cudaMemcpy(sumCost, GsumCost, ISLAND*POPULATION*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(minCost, GminCost, ISLAND*2*sizeof(float), cudaMemcpyDeviceToHost);

  // printf("\n");
	// for(int i=0;i<ISLAND;i++){
	// 	printf("第%d島嶼: \n", i);
	// 	for(int p=0;p<POPULATION;p++){
	// 		printf("%d: ", p);
	// 	  printf("sum = %f", sumCost[i][p]);
	// 		printf("\n");
	// 	}
	// }


  int data2[ISLAND][POPULATION][FACILITY]; // facility
  int *Gdata2;
  cudaMalloc((void**)&Gdata2, ISLAND*POPULATION*FACILITY*sizeof(int));
	short int bay2[ISLAND][POPULATION][FACILITY]; //bay
  short int *Gbay2;
  cudaMalloc((void**)&Gbay2, ISLAND*POPULATION*FACILITY*sizeof(short int));

	float probability[ISLAND][POPULATION] = {0.0}; // �U�Ӿ��v

  // for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
	// 		printf("pro%f \n", probability[i][p]);
	// 	}
	// }
  float *Gprobability;
  cudaMalloc((void**)&Gprobability, ISLAND*POPULATION*sizeof(float));

	float totalPro[ISLAND] = {0.0};                // �`(�����˼�)
  float *GtotalPro;
  cudaMalloc((void**)&GtotalPro, ISLAND*sizeof(float));

  for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			totalPro[i] = totalPro[i] + (1.0 / sumCost[i][p]);
			// printf("%f %f\n", totalPro[i], (1.0 / sumCost[i][p]));
		}
	}

  cudaMemcpy(GtotalPro, totalPro, ISLAND*sizeof(float), cudaMemcpyHostToDevice);


	// for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
	// 		printf("%f %f\n", totalPro[i], (1.0 / sumCost[i][p]));
	// 	}
	// }

  calProbability<<<ISLAND, POPULATION>>>(Gprobability, GtotalPro, GsumCost);

  cudaMemcpy(probability, Gprobability, ISLAND*POPULATION*sizeof(float), cudaMemcpyDeviceToHost);

	// for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
	// 		printf("%f %f %f \n", probability[i][p], (1.0 / sumCost[i][p]), totalPro[i]);
	// 	}
	// }


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
	// printf("probability2\n");
	// for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
	// 		printf("%f ", probability2[i][p]);
	// 	}
	// }



  int *Gtem, *Gtem2, *Gyes, *Gsss;// choose two to crossover and if yes or not and choose area
  int tem[ISLAND*POPULATION], tem2[ISLAND*POPULATION], yes[ISLAND*POPULATION], sss[ISLAND*POPULATION];
  cudaMalloc((void**)&Gtem, ISLAND*POPULATION*sizeof(int));
  cudaMalloc((void**)&Gtem2, ISLAND*POPULATION*sizeof(int));
  cudaMalloc((void**)&Gyes, ISLAND*POPULATION*sizeof(int));
  cudaMalloc((void**)&Gsss, ISLAND*POPULATION*sizeof(int));

  int *GmutaYes, *GmutaTem, *GmutaTem2;
  int mutaYes[ISLAND*POPULATION], mutaTem[ISLAND*POPULATION], mutaTem2[ISLAND*POPULATION];
  cudaMalloc((void**)&GmutaYes, ISLAND*POPULATION*sizeof(int));
  cudaMalloc((void**)&GmutaTem, ISLAND*POPULATION*sizeof(int));
  cudaMalloc((void**)&GmutaTem2, ISLAND*POPULATION*sizeof(int));
  for(int i=0;i<ISLAND*POPULATION;i++){
    tem[i] = rand(); // first change
    tem2[i] = rand(); // second change
    yes[i] = rand(); // crossover or not
    sss[i] = rand(); // bay to crossover
    mutaYes[i] = rand(); // mutation or not
    mutaTem[i] = rand(); // first to change
    mutaTem2[i] = rand(); // second to change
    // printf("%d %d %d %d %d %d %d\n", tem[i], tem2[i], yes[i], sss[i], mutaYes[i], mutaTem[i], mutaTem2[i]);
  }

  cudaMemcpy(Gtem, tem, ISLAND*POPULATION*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(Gtem2, tem2, ISLAND*POPULATION*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(Gyes, yes, ISLAND*POPULATION*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(Gsss, sss, ISLAND*POPULATION*sizeof(int), cudaMemcpyHostToDevice);


  cudaMemcpy(GmutaYes, mutaYes, ISLAND*POPULATION*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(GmutaTem, mutaTem, ISLAND*POPULATION*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(GmutaTem2, mutaTem2, ISLAND*POPULATION*sizeof(int), cudaMemcpyHostToDevice);

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
  cudaMemcpy(tem, Gtem, ISLAND*POPULATION*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(tem2, Gtem2, ISLAND*POPULATION*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(yes, Gyes, ISLAND*POPULATION*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(sss, Gsss, ISLAND*POPULATION*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(count, Gcount, 10*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(getP, GetP, 10*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(getP2, GetP2, 10*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(test, Gtest, 10*sizeof(int), cudaMemcpyDeviceToHost);

  // mutation facility
	printf("\nready to mutation\n");

  mutation<<<ISLAND, POPULATION>>>(Gdata2, GmutaYes, GmutaTem, GmutaTem2);

  cudaMemcpy(mutaYes, GmutaYes, ISLAND*POPULATION*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(mutaTem, GmutaTem, ISLAND*POPULATION*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(mutaTem2, GmutaTem2, ISLAND*POPULATION*sizeof(int), cudaMemcpyDeviceToHost);

  int *GmutaBayYes, *GmutaBayTem;
  int mutaBayYes[ISLAND*POPULATION], mutaBayTem[ISLAND*POPULATION];
  cudaMalloc((void**)&GmutaBayYes, ISLAND*POPULATION*sizeof(int));
  cudaMalloc((void**)&GmutaBayTem, ISLAND*POPULATION*sizeof(int));

  cudaMemcpy(GmutaBayYes, mutaBayYes, ISLAND*POPULATION*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(GmutaBayTem, mutaBayTem, ISLAND*POPULATION*sizeof(int), cudaMemcpyHostToDevice);

  mutationBay<<<ISLAND, POPULATION>>>(Gbay2, GmutaBayYes, GmutaBayTem);

  cudaMemcpy(mutaBayYes, GmutaBayYes, ISLAND*POPULATION*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(mutaBayTem, GmutaBayTem, ISLAND*POPULATION*sizeof(int), cudaMemcpyDeviceToHost);

  // migration
	if( (gggggg+1) % MIGRATION == 0 && (gggggg+1) != 0 && ISLAND > 1){
		printf("***migration***\n");

		int temp3[ISLAND][POPULATION/2][FACILITY];
		short temp4[ISLAND][POPULATION/2][FACILITY-1];
		int indexCost[ISLAND][POPULATION];

		for(int i=0;i<ISLAND;i++){
			for(int p=0;p<POPULATION;p++){
				indexCost[i][p] = p;
			}
		}

		// bubble sort
		float temp;
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
					int p = indexCost[i][k];
					for(int f=0;f<FACILITY;f++){
						data2[i][frontP][f] = temp3[ISLAND-1][backP][f];
					}
					for(int f=0;f<FACILITY-1;f++){
						bay2[i][frontP][f] = temp4[ISLAND-1][backP][f];
					}
				}
			} // else end

		} // for end




		} // if migration end


  // printf("count: \n");
  // for(int i=0;i<10;i++){
  //   printf("%d ", count[i]);
  // }

  // printf("\nget: \n");
  // for(int i=0;i<10;i++){
  //   printf("%d %d\n", getP[i], getP2[i]);
  // }

  // printf("\ntest: \n");
  // for(int i=0;i<10;i++){
  //   printf("%f\n", test[i]);
  // }

  // printf("\nTEM: \n");
  // for(int i=0;i<20;i++){
  //   printf("%d %d %d %d\n", tem[i], tem2[i], yes[i], sss[i]);
  // }
  //
  // printf("\nmutation: \n");
  // for(int i=0;i<20;i++){
  //   printf("%d %d %d\n", mutaYes[i], mutaTem[i], mutaTem2[i]);
  // }
  float answer[3];
  answer[0] = 0;
  answer[1] = 0;
  answer[2] = sumCost[0][0];
  for(int i=0;i<ISLAND;i++){
		// printf("第%d島嶼(OF): \n", i);
		for(int p=0;p<POPULATION;p++){
			// printf("%f ", sumCost[i][p]);
      if(sumCost[i][p] < answer[2]){
        answer[0] = i;
        answer[1] = p;
        answer[2] = sumCost[i][p];
      }
			// printf("\n");
		}
	}


  printf("最小: %.0f %.0f = %f\n", answer[0], answer[1], answer[2]);



  // for(int i=0;i<ISLAND;i++){
  //   for(int p=0;p<POPULATION;p++){
  //     printf("\n交配結果(data2)%d\n", p);
  //     for(int f=0;f<FACILITY;f++){
  //       printf("%d ", data2[i][p][f]);
  //     }
  //     printf("\n");
  //   }
  // }

  // parent to child
	// printf("***chile to parent!!!***\n");
	for(int i=0;i<ISLAND;i++){
		// printf("island%d\n", i);
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY;f++){
				data[i][p][f] = data2[i][p][f];
				// printf("%d ", data[i][p][f]);
			}
			// printf("\n");
		}
	}

  // 子代BAY
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY-1;f++){
				bay[i][p][f] = bay2[i][p][f];
			}
		}
	}

  cudaFree(GA);
  cudaFree(GB);
  cudaFree(Gdata2);
  cudaFree(Gbay2);
  cudaFree(GsumCost);
  cudaFree(GminCost);
  cudaFree(GtotalCost);
  cudaFree(Gtem);
  cudaFree(Gtem2);
  cudaFree(GetP);
  cudaFree(GetP2);
  cudaFree(Gtest);
  cudaFree(Gprobability);
  cudaFree(Gprobability2);
  cudaFree(Gyes);
  cudaFree(Gsss);
  cudaFree(GmutaYes);
  cudaFree(GmutaTem);
  cudaFree(GmutaTem2);
  cudaFree(Gdistance);
  cudaFree(Gposition);
  } // GENERATION 結束

  END = clock();
  printf("程式執行所花費： %lf S\n", (double)clock()/CLOCKS_PER_SEC);
  printf("進行運算所花費的時間： %lf S\n", (END - START) / CLOCKS_PER_SEC);
  // cout << endl << "程式執行所花費：" << (double)clock()/CLOCKS_PER_SEC << " S" ;
  // cout << endl << "進行運算所花費的時間：" << (END - START) / CLOCKS_PER_SEC << " S" << endl;
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
