#include <iostream>
#include <stdio.h>


#define INIT 200
#define NUMBER 5

using namespace std;

int a[NUMBER];
bool used[NUMBER];

int data[INIT][NUMBER];
int count = 0;

float width[NUMBER];

 
void print(int N)
{   

    for (int i=0; i<N; ++i){
        data[count][i] = a[i] + 1;        
    }
    count++;
}
 
void backtrack(int n, int N)
{
    if (n == N) {print(N); return;}
 
    for (int i=0; i<N; i++)
        if (!used[i])
        {
            used[i] = true;
            a[n] = i;
            backtrack(n+1, N);
            used[i] = false;
        }
}
 
void enumerate_permutations(int N)
{
    for (int i=0; i<N; i++) used[i] = false;
    backtrack(0, N);
}
void width_init(){
    width[0] = 9;
    width[1] = 3;
    width[2] = 7;
    width[3] = 9;
    width[4] = 1;
}

int main(void){
    width_init();
    enumerate_permutations(NUMBER);
    cout << "\n\n";

    // for(int i=0;i<count;i++){
    //     for(int j=0;j<NUMBER;j++){
    //         cout << data[i][j];
    //     }
    //     cout << "\n";
    // }



    // read the file cost
    FILE *fPtr;

    int cost_num = NUMBER * (NUMBER - 1) / 2;

	fPtr=fopen("cost.txt","r");
	int cost[NUMBER][NUMBER] = {0};
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
    float position[INIT][NUMBER];
    
    for(int i=0;i<count;i++){
        position[i][0] = width[0] / 2;
    }
    for(int c=0;c<count;c++){
        for(int i=1;i<NUMBER;i++){
            position[c][i] = width[i] / 2 + position[c][i-1] + width[i-1] / 2;
        }
    }

    
    

    // OF
    float OF[INIT][cost_num] ;
    
    for(int c=0;c<count;c++){
        int count2 = 0;
        for(int i=0;i<NUMBER-1;i++){
            for(int j=i+1;j<NUMBER;j++){
                OF[c][count2] = ( position[c][j] - position[c][i] ) * cost[ data[c][i] - 1 ][ data[c][j] - 1 ];
                // printf("%f %f\n", position[c][i] , position[c][j]);
                // printf("%d %d %d %f %d\n", data[c][i]-1, data[c][j] - 1, cost[ data[c][i] - 1 ][ data[c][j] - 1 ], position[c][j] - position[c][i] , count2);
                count2++;
            }
        }
    }

    // printf("%f %f\n", position[0][0] , position[0][1]);
    // printf("%d %d %d %f %d\n", data[c][i]-1, data[c][j] - 1, cost[ data[c][i] - 1 ][ data[c][j] - 1 ], position[c][j] - position[c][i] , count2);
    // printf("%f %f\n", OF[0][0] , OF[0][1]);

    float total_OF[INIT] = {0.0} ;
    for(int c=0;c<count;c++){
        for(int i=0;i<cost_num;i++){
            total_OF[c] += OF[c][i];
        }
    }    

    for(int c=0;c<count;c++){
        printf("OF = %f\n", total_OF[c]);
    }
    


    return 0;
}