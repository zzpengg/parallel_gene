#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM 5
#define RANGE 10

int main(void){

    double START,END;
    START = clock();
    srand(time(NULL));

    int data[NUM];


    // generate number
    for(int i=0;i<NUM;i++){
        data[i] = i;
    }
    
    // shuffle
    for(int i = 0; i < NUM; i++) {
        int j = rand() % NUM;
        int tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }

    float width[NUM];
    for(int i=0;i<NUM;i++){
        width[i] = rand() % RANGE;
    }

    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
    data[3] = 4;
    data[4] = 5;

    width[0] = 9;
    width[1] = 3;
    width[2] = 7;
    width[3] = 9;
    width[4] = 1;


    
    // read the file cost
    FILE *fPtr;

    int cost_num = NUM * (NUM - 1) / 2;

	fPtr=fopen("cost.txt","r");
	int cost[NUM][NUM] = {0};
	int temp[cost_num][3]; // cost
	for(int i=0;i<cost_num;i++){
		fscanf(fPtr , "%d %d %d" , &temp[i][0], &temp[i][1], &temp[i][2]);
	}
	fclose(fPtr);

    for(int i=0;i<cost_num;i++){ // 2 dimention cost
		cost[ temp[i][0]-1 ][ temp[i][1]-1] = temp[i][2];
        cost[ temp[i][1]-1 ][ temp[i][0]-1] = temp[i][2];
	}

    // cal position
    float position[NUM];
    position[0] = width[0] / 2;
    for(int i=1;i<NUM;i++){
        position[i] = width[i] / 2 + position[i-1] + width[i-1] / 2;
    }

    // OF
    float OF[cost_num];
    int count = 0;
    for(int i=0;i<NUM-1;i++){
        for(int j=i+1;j<NUM;j++){
            OF[count] = ( position[j] - position[i] ) * cost[ data[i] - 1 ][ data[j] - 1 ];
            printf("%f %f\n", position[i] , position[j]);
            printf("%d %d %d %f %d\n", data[i]-1, data[j] - 1, cost[ data[i] - 1 ][ data[j] - 1 ], position[j] - position[i] , count);
            count++;
        }
    }

    float total_OF = 0.0 ;
    for(int i=0;i<cost_num;i++){
        total_OF += OF[i];
    }


    for(int i=0;i<NUM;i++){
        printf("%d ", data[i]);
    }
    printf("\n");
    for(int i=0;i<NUM;i++){
        printf("%f ", width[i]);
    }

    printf("OF = %f\n", total_OF);

    

    END = clock();

    printf("time = %f\n", (END - START) / CLOCKS_PER_SEC);
    return 0;
}