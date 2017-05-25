#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ISLAND 10 
#define POPULATION 50
#define FACILITY 20 
#define GENERATION 70 
#define CROSSOVER 0.6
#define MUTATION 0.03
#define MIGRATION 15
#define INDIVIDUAL 5

#define H 3 // BAY height
#define W 2 // BAY width

void shuffle(int* facility);

int main(){

	srand(time(NULL));

	int data[ISLAND][POPULATION][FACILITY]; // facility
	short int bay[ISLAND][POPULATION][FACILITY-1]; //bay

	int facility[FACILITY];


  for(int i=0;i<ISLAND;i++){ // shuffle the sorted facility
//		printf("new island%d\n", i);
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
//				printf("%d ", data[i][p][f]);
			}
//			printf("\n");
			for(int b=0;b<FACILITY-1;b++){
				int j = rand() % 2;
		    bay[i][p][b] = j;
			}
		}
	}
  // read ther cost
	FILE *fPtr;
	
	int ttt = FACILITY * (FACILITY-1) / 2;

	fPtr=fopen("cost.txt","r");
	int cost[FACILITY][FACILITY] = {0};
	int temp[ttt][3]; // cost
	for(int i=0;i<ttt;i++){
		for(int a=0;a<3;a++){
			fscanf(fPtr , "%d " , &temp[i][a]);
		}
	}
	fclose(fPtr);
	for(int i=0;i<ttt;i++){ // 2 dimention cost
		cost[temp[i][0]-1][temp[i][1]-1] = temp[i][2];
	}

  // print COST
	// for(int i=0;i<15;i++){
	// 	for(int j=0;j<3;j++){
	// 		printf("%d ", temp[i][j]);
	// 	}
	// 	printf("\n");
	// }
	// for(int i=0;i<FACILITY;i++){
	// 	for(int j=0;j<FACILITY;j++){
	// 		printf("%d ", cost[i][j]);
	// 	}
	// 	printf("\n");
	// }

	// print on final txt
	FILE *FIN;

	FIN = fopen("final.txt","w");


	for(int g=0;g<GENERATION;g++){
	printf("generation = %d\n", g);
	fprintf(FIN, "第%d世代\n", g);


	// calculate position
	// printf("\nready to calculate position\n");
	float position[ISLAND][POPULATION][FACILITY][2];
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			int len = 1;
			int next = 0;
			for(int f=0;f<FACILITY;f++){
				if(bay[i][p][f] == 0){
					len = len + 1;
				}
				if(bay[i][p][f] == 1 || f == FACILITY - 1 ){
					if(f == FACILITY - 1 && bay[i][p][f] == 0){
						len = len - 1;
					}
					float x = W / 2.0 + next;

					for(int j=0;j<len;j++){

						position[i][p][f+j-len+1][0] = x;

						float y = H / (len * 2.0) * ( (j * 2) + 1) ;

						position[i][p][f+j-len+1][1] = y;
					}
					len = 1;

					next = next + W;
				}
			}
		}
	}


  // print position
	// for(int i=0;i<ISLAND;i++){
	// 	printf("island%d ", i);
	// 	for(int p=0;p<POPULATION;p++){
	// 		printf("po%d = ",p);
	// 		for(int f=0;f<FACILITY;f++){
	// 			for(int k=0;k<3;k++){
	// 				printf("%f ", position[i][p][f][k]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	// }

	// calculate distance
	// printf("\nready to calculate distance\n");
	float distance[ISLAND][POPULATION][FACILITY][FACILITY] = {0};

	for(int i=0;i<ISLAND;i++){
		// printf("\ndistance island%d\n", i);
		for(int p=0;p<POPULATION;p++){
			// printf("\ndistance population%d\n", p);
			for(int f=0;f<FACILITY;f++){
				// printf("\ndistance calculate facility%d\n", f);
				for(int j=f+1;j<FACILITY;j++){

					float x1 = position[i][p][f][0];
					float y1 = position[i][p][f][1];

					int x = data[i][p][f];
					// printf("x = %d\n", x);
					float x2 = position[i][p][j][0];
					float y2 = position[i][p][j][1];
					int y = data[i][p][j];
					// printf("y= %d\n", y);
					if(y2 > y1){
						distance[i][p][x][y] = sqrt( (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ) ;
						distance[i][p][y][x] = sqrt( (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ) ;
					}
					else{
						distance[i][p][x][y] = sqrt( (x2 - x1) * (x2 - x1) + (y1 - y2) * (y1 - y2) ) ;
						distance[i][p][y][x] = sqrt( (x2 - x1) * (x2 - x1) + (y1 - y2) * (y1 - y2) ) ;
					}
				}
			}
		}
	}
	// printf("\ncalculate distance end\n");

  // print distance
	// for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
	// 		for(int f=0;f<FACILITY;f++){
	// 			for(int j=0;j<FACILITY;j++){
	// 				printf("%f ", distance[i][p][f][j]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	// }

	// calculate totalcost
	// printf("\nready to calculate totalcost\n");
	float totalCost[ISLAND][POPULATION][FACILITY][FACILITY] = {0.0};

	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY;f++){
				for(int j=0;j<FACILITY;j++){
					totalCost[i][p][f][j] = cost[f][j] * distance[i][p][f][j];
				}
			}
		}
	}

  // print totalCost
	// for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
	// 		for(int f=0;f<FACILITY;f++){
	// 			for(int j=0;j<FACILITY;j++){
	// 				printf("%f ", totalCost[i][p][f][j]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	// }

	// calculate OF

	float sumCost[ISLAND][POPULATION]={0.0};
	float minCost[ISLAND][2];
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY;f++){
				for(int j=0;j<FACILITY;j++){
					sumCost[i][p] += totalCost[i][p][f][j];
				}
			}
			if(p==0){
				minCost[i][0] = sumCost[i][0];
				minCost[i][1] = 0;
			}else if(minCost[i][0] > sumCost[i][p]){
				minCost[i][0] = sumCost[i][p];
				minCost[i][1] = p;
			}
		}
	}

	// parent
	fprintf(FIN, "\n");
	for(int i=0;i<ISLAND;i++){
		fprintf(FIN, "第%d島嶼: \n", i);
		for(int p=0;p<POPULATION;p++){
			fprintf(FIN, "%d: ", p);
			for(int f=0;f<FACILITY;f++){
				fprintf(FIN, "%d ", data[i][p][f]);
			}
			fprintf(FIN, "\n");
			fprintf(FIN, " : ");
			for(int f=0;f<FACILITY-1;f++){
				fprintf(FIN, "%d ", bay[i][p][f]);
			}
			fprintf(FIN, "sum = %f", sumCost[i][p]);
			fprintf(FIN, "\n");
		}
	}

	// ���t

	// �U�@�Ӥl�N
	int data2[ISLAND][POPULATION][FACILITY]; // facility
	short int bay2[ISLAND+10][POPULATION][FACILITY]; //bay

	float probability[ISLAND][POPULATION] = {0.0}; // �U�Ӿ��v
	float totalPro[ISLAND] = {0.0};                // �`(�����˼�)

	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			totalPro[i] = totalPro[i] + (1.0 / sumCost[i][p]);
//			printf("%f %f\n", totalPro[i], (1.0 / sumCost[i][p]));
		}
	}

	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			probability[i][p] = (1.0 / sumCost[i][p]) / totalPro[i] ;
//			printf("%f %f %f \n", probability[i][p], (1.0 / sumCost[i][p]), totalPro[i]);
		}
	}

	// print sumCost
	// for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
	// 		printf("%f ", sumCost[i][p]);
	// 	}
	// }

	// print probability
	// for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
	// 		printf("%f ", probability[i][p]);
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

	// print probability2 (Roulette)
//	printf("probability2\n");
//	for(int i=0;i<ISLAND;i++){
//		for(int p=0;p<POPULATION;p++){
//			printf("%f ", probability2[i][p]);
//		}
//	}
	int num=0;
	printf("\nwill in crossover\n");
	for(int i=0;i<ISLAND;i++){
		printf("\ncrossver island = %d\n", i);
		fprintf(FIN, "第%d島嶼: \n", i);
		num = 0;
		for(int n=0;n<POPULATION/2;n++){
			// �D���ǭǥ��t
			int tem = rand() % 100;
			float get = tem * 0.01;
//			printf("get = %f\n", get);
			int getP = 0;
			float tem2 = rand() % 100;
			float get2 = tem2 * 0.01;
			int getP2 = 0;
			// printf("get = %f %f\n", get, get2);
			for(int p=0;p<POPULATION-1;p++){
				if(get >= probability2[i][p] && get < probability2[i][p+1]){
					getP = p+1;
//					printf("getP = %d\n", getP);
					break;
				}
				else if(p==POPULATION-2){
					getP = p+1;
					break;
				}
			}
			for(int p=0;p<POPULATION-1;p++){
				if(get2 >= probability2[i][p] && get2 < probability2[i][p+1]){
					getP2 = p+1;
					break;
				}
				else if(p==POPULATION-2){
					getP2 = p+1;
					break;
				}
			}
			fprintf(FIN, "取的資料: ");
			fprintf(FIN, "%d %d\n", getP, getP2);
			// printf("new data2: \n");
			for(int f=0;f<FACILITY;f++){
				data2[i][num][f] = data[i][getP][f];
				bay2[i][num][f] = bay[i][getP][f];
				// printf("%d %d %d = %d\n", i, num, f, data2[i][num][f]);
				fprintf(FIN, "%d ", data2[i][num][f]);
			}
			// printf("\n");

			fprintf(FIN, "\t");
			num++;
			for(int f=0;f<FACILITY;f++){
				data2[i][num][f] = data[i][getP2][f];
				bay2[i][num][f] = bay[i][getP2][f];
//				printf("%d ", data2[i][num][f]);
				fprintf(FIN, "%d ", data2[i][num][f]);
			}
//			printf("\n");
			fprintf(FIN, "\n");
			num++;

			// �O�_���t
			int t = rand() % 100;
			float yes = t * 0.01;
			// fprintf(FIN, "取得%f \n", yes);
			if(yes <= CROSSOVER){
				fprintf(FIN, "交配\n");
				// ���tfacility
				int sss = FACILITY - 1;
				int seq = rand() % sss;
				fprintf(FIN, "選第%d個交配\n", seq);
				// num-1 === getP2
				// num-2 === getP

				int cross[4][2];

				cross[0][0] = data2[i][num-2][seq];
				cross[0][1] = data2[i][num-1][seq];
				cross[1][0] = data2[i][num-2][seq];
				cross[1][1] = data2[i][num-1][seq+1];
				cross[2][0] = data2[i][num-2][seq+1];
				cross[2][1] = data2[i][num-1][seq];
				cross[3][0] = data2[i][num-2][seq+1];
				cross[3][1] = data2[i][num-1][seq+1];

				for(int c=0;c<4;c++){
					fprintf(FIN, "%d %d\n", cross[c][0], cross[c][1]);
//					printf("%d %d\n", cross[c][0], cross[c][1]);
				}


				int temp = data2[i][num-1][seq];
				// fprintf(FIN, "temp = %d ", temp);
				int temp2 = data2[i][num-1][seq+1];
				// fprintf(FIN, "temp2 = %d ", temp2);
				data2[i][num-1][seq] = data2[i][num-2][seq];
				// fprintf(FIN, "temp3 = %d ", data2[i][num-2][seq]);
				data2[i][num-1][seq+1] = data2[i][num-2][seq+1];
				// fprintf(FIN, "temp4 = %d ", data2[i][num-2][seq+1]);
				data2[i][num-2][seq] = temp;
				data2[i][num-2][seq+1] = temp2;

				fprintf(FIN, "第一個交換結果\n");
				for(int c=0;c<FACILITY;c++){
					fprintf(FIN, "%d ", data2[i][num-2][c]);
				}

				fprintf(FIN, "\n第二個交換結果\n");
				for(int c=0;c<FACILITY;c++){
					fprintf(FIN, "%d ", data2[i][num-1][c]);
				}

				int count = 0;// calculate how much the same
				for(int c=0;c<4;c++){
					if(cross[c][0] == cross[c][1]){
						count++;
					}
				}

				printf("\ncrossover!!\n");
				fprintf(FIN, "有%d個一樣\n", count);
				switch (count) {
					case 0: // �������@��
						// ����getP
						fprintf(FIN, "\n產生第一個子代: ");
						for(int c=0;c<FACILITY;c++){
							if(c != seq){
								if(data2[i][num-2][c] == cross[0][1]){
									data2[i][num-2][c] = cross[0][0];
								}
								if(data2[i][num-2][c] == cross[3][1]){
									data2[i][num-2][c] = cross[3][0];
								}
							}
							else{
								fprintf(FIN, "%d ", data2[i][num-2][c]);
								c++;
							}
							fprintf(FIN, "%d ", data2[i][num-2][c]);
						}
						// �A��getP2
						fprintf(FIN, "\n產生第二個子代: ");
						for(int c=0;c<FACILITY;c++){
							if(c != seq){
								if(data2[i][num-1][c] == cross[0][0]){
									data2[i][num-1][c] = cross[0][1];
								}
								if(data2[i][num-1][c] == cross[3][0]){
									data2[i][num-1][c] = cross[3][1];
								}
							}
							else{
//								printf("%d ", data2[i][num-1][c]);
								fprintf(FIN, "%d ", data2[i][num-1][c]);
								c++;
							}
//							printf("%d ", data2[i][num-1][c]);
							fprintf(FIN, "%d ", data2[i][num-1][c]);
						}
						break;
					case 1: // only one the same
						temp = 99;
						fprintf(FIN, "\n產生子代: ");
						for(int c=0;c<4;c++){
							if(cross[c][0] == cross[c][1]){
								temp = cross[c][0];
							}
						}
						// deal with num-2
						for(int c=0;c<4;c++){
							if(cross[c][0] != temp && cross[c][1] != temp){
								for(int f=0;f<FACILITY;f++){
									if(f != seq){
										if(data2[i][num-2][f] == cross[c][1]){
											data2[i][num-2][f] = cross[c][0];
										}
									}
									else{
										f++;
									}
								}
							}
						}
						// deal with num-1
						for(int c=0;c<4;c++){
							if(cross[c][0] != temp && cross[c][1] != temp){
								for(int f=0;f<FACILITY;f++){
									if(f != seq){
										if(data2[i][num-1][f] == cross[c][0]){
											data2[i][num-1][f] = cross[c][1];
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

				fprintf(FIN, "\n交配結果\n");
				for(int f=0;f<FACILITY;f++){
//					printf("%d ", data2[i][num-2][f]);
					fprintf(FIN, "%d ", data2[i][num-2][f]);
				}
				fprintf(FIN, "\t");
//				printf("\t");
				for(int f=0;f<FACILITY;f++){
//					printf("%d ", data2[i][num-1][f]);
					fprintf(FIN, "%d ", data2[i][num-1][f]);
				}
				fprintf(FIN, "\n");


				// ���tbay
				temp = bay2[i][num-1][seq];
				temp2 = bay2[i][num-1][seq+1];
				bay2[i][num-1][seq] = bay2[i][num-2][seq];
				bay2[i][num-1][seq+1] = bay2[i][num-2][seq+1];
				bay2[i][num-2][seq] = bay2[i][num-1][seq];
				bay2[i][num-2][seq+1] = bay2[i][num-1][seq+1];
			}else {
				fprintf(FIN, "沒有交配\n");
			}
			printf("\ncrossover end\n");
		} // population end
	} // island end

//	printf("***crossover end***\n");
//	printf("data2[1][0][0] = %d\n", data2[1][0][0]);
//	for(int i=0;i<ISLAND;i++){
//		printf("island%d\n", i);
//		for(int p=0;p<POPULATION;p++){
//			for(int f=0;f<FACILITY;f++){
//				printf("%d ", data2[i][p][f]);
//			}
//			printf("\n");
//		}
//	}

	// new child
	// printf("新的子代\n");
	// for(int i=0;i<ISLAND;i++){
	// 	for(int p=0;p<POPULATION;p++){
	// 		for(int f=0;f<FACILITY;f++){
	// 			printf("%d ", data2[i][p][f]);
	// 		}
	// 		printf("\n");
	// 	}
	// 	printf("\n");
	// }

	// mutation facility
	printf("\nready to mutation\n");
	for(int i=0;i<ISLAND;i++){
		printf("\nmutation%d\n", i);
		for(int p=0;p<POPULATION;p++){
			int t = rand() % 100;
			float yes = t * 0.01;
			// fprintf(FIN, "取得%f \n", yes);
			if(yes < MUTATION){
				// fprintf(FIN, "第%d突變\n", p);
				int get = rand() % FACILITY;
				int get2 = rand() % FACILITY;
				int temp = data2[i][p][get];
				data2[i][p][get] = data2[i][p][get2];
				data2[i][p][get2] = temp;
			}else {
				// fprintf(FIN, "%d沒有突變\n", p);
			}
		}
	}

	// mutation bay
	printf("\nready to mutation bay\n");
	for(int i=0;i<ISLAND;i++){
		printf("mutation bay = %d\n", i);
		for(int p=0;p<POPULATION;p++){
			int t = rand() % 100;
			float yes = t * 0.01;
			if(yes < MUTATION){
				int get = rand() % FACILITY;
				if(bay2[i][p][get] == 0){
					bay2[i][p][get] = 1;
				}else {
					bay2[i][p][get] = 0;
				}
			}
		}
	}

	// migration
	if( (g+1) % MIGRATION == 0 && (g+1) != 0 && ISLAND > 1){
		printf("***migration***\n");
		fprintf(FIN, "***migration***\n");

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
//		for(int i=0;i<ISLAND;i++){
//			for(int p=0;p<POPULATION;p++){
//				printf("%d ", indexCost[i][p]);
//			}
//			printf("\n");
//		}

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



	// 印出距離
	// for(int i=0;i<ISLAND;i++){
	// 	fprintf(FIN, "第%d島嶼: \n", i);
	// 	for(int p=0;p<POPULATION;p++){
	// 		fprintf(FIN, "%d: \n", p);
	// 		for(int f=0;f<FACILITY;f++){
	// 			for(int j=0;j<FACILITY;j++){
	// 				fprintf(FIN, "%f ", distance[i][p][f][j]);
	// 			}
	// 			fprintf(FIN, "\n");
	// 		}
	// 	}
	// }


	// 印出 固定成本
	// fprintf(FIN, "成本\n");
	// for(int f=0;f<FACILITY;f++){
	// 	for(int j=0;j<FACILITY;j++){
	// 		fprintf(FIN, "%d ", cost[f][j]);
	// 	}
	// 	fprintf(FIN, "\n");
	// }

	// 印出 成本
	// for(int i=0;i<ISLAND;i++){
	// 	fprintf(FIN, "第%d島嶼: \n", i);
	// 	for(int p=0;p<POPULATION;p++){
	// 		fprintf(FIN, "%d: \n", p);
	// 		for(int f=0;f<FACILITY;f++){
	// 			for(int j=0;j<FACILITY;j++){
	// 				fprintf(FIN, "%f ", totalCost[i][p][f][j]);
	// 			}
	// 			fprintf(FIN, "\n");
	// 		}
	// 	}
	// }
	
	float minnn = sumCost[0][0];

	for(int i=0;i<ISLAND;i++){
		fprintf(FIN, "第%d島嶼(OF): \n", i);
		for(int p=0;p<POPULATION;p++){
			if(sumCost[i][p] < minnn){
				minnn = sumCost[i][p];
			}
			fprintf(FIN, "%f ", sumCost[i][p]);
			fprintf(FIN, "\n");
		}
	}

	fprintf(FIN, "新的子代\n");
	for(int i=0;i<ISLAND;i++){
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY;f++){
				fprintf(FIN, "%d ", data2[i][p][f]);
			}
			fprintf(FIN, "\n");
		}
		fprintf(FIN, "\n");
	}

	// parent to child
	printf("***chile to parent!!!***\n");
	for(int i=0;i<ISLAND;i++){
		printf("island%d\n", i);
		for(int p=0;p<POPULATION;p++){
			for(int f=0;f<FACILITY;f++){
				data[i][p][f] = data2[i][p][f];
//				printf("%d ", data[i][p][f]);
			}
//			printf("\n");
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

	printf("minnn = %f", minnn);




} // GENERATION 結束

	

	fclose(FIN);

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
