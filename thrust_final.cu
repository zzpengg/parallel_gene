#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <iostream>

#define ISLAND 2
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

int main(void){

	srand(time(NULL));

  int data[ISLAND][POPULATION][FACILITY]; // facility
	short int bay[ISLAND][POPULATION][FACILITY-1]; //bay

  thrust::device_vector<int> dev_data(ISLAND*POPULATION*FACILITY);
  // thrust::device_vector<short int> bay[ISLAND][POPULATION][FACILITY];

  // ***data***

  // int ***data = NULL;
  // data = (int ***)malloc(sizeof(int **)*ISLAND);
  // for(int i=0;i<ISLAND;i++){
  //   data[i] = (int **)malloc(sizeof(int *)*POPULATION);
  // }
  //
  // for(int i=0;i<ISLAND;i++){
  //   for(int p=0;p<POPULATION;p++){
  //     data[i][j] = (int *)malloc(sizeof(int)*FACILITY);
  //   }
  // }
  // ***data end***


	int facility[FACILITY];


  for(int i=0;i<ISLAND;i++){ // shuffle the sorted facility
		printf("new island = %d\n", i);
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
				std::cout << data[i][p][f] << " ";
			}
			std::cout << "\n" ;
			for(int b=0;b<FACILITY-1;b++){
				int j = rand() % 2;
		    bay[i][p][b] = j;
			}
		}
	}

	// throw to dev_data device memory
	thrust::copy(&(data[0][0][0]), &(data[ISLAND-1][POPULATION-1][FACILITY-1]), dev_data.begin());
  // read ther cost
	FILE *fPtr;

	fPtr=fopen("cost.txt","r");
	int cost[FACILITY][FACILITY] = {0};

	int temp[15][3]; // cost
	for(int i=0;i<15;i++){
		for(int a=0;a<3;a++){
			fscanf(fPtr , "%d " , &temp[i][a]);
		}
	}
	fclose(fPtr);
	std::cout << "cost: " << std::endl;
	for(int i=0;i<15;i++){ // 2 dimention cost
		cost[temp[i][0]-1][temp[i][1]-1] = temp[i][2];
    std::cout << temp[i][0] << " " << temp[i][1] << " " << temp[i][2] << std::endl;
	}


	std::cout << "\ndevice data: " << std::endl;
  for(int i = 0; i < ISLAND; i++){
		std::cout << "island: " << i << std::endl;
    for(int p=0;p<POPULATION;p++){
      for(int f=0;f<FACILITY;f++){
        std::cout << dev_data[i+p+f] << " ";
      }
			std::cout << std::endl;
    }
  }

	for(int g=0;g<GENERATION;g++){
		printf("generation = %d\n", g);

		thrust::device_vector<float> position;

	} // generation loop end

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
