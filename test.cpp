#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LENGTH 8

int main(){
	char ch[LENGTH] = {'A','B','C','D','E','F','G','H'};
	int flex[LENGTH-1] = {1,0,1,0,0,1,0};
	float cost[LENGTH][LENGTH] = {
	 {1,1,1,1,1,1,1,1},
	 {1,1,1,1,1,1,1,1},
	 {1,1,1,1,1,1,1,1},
	 {1,1,1,1,1,1,1,1},
	 {1,1,1,1,1,1,1,1},
	 {1,1,1,1,1,1,1,1},
	 {1,1,1,1,1,1,1,1},
	 {1,1,1,1,1,1,1,1},
	};
	/*
	A
	B
	C
	D
	E
	F
	G
	H
	*/
	float position[LENGTH][2];
//	printf("cost = %f\n",cost[0][0]);
	
	int H = 2;
	int W = 1;
	
	int len = 1;
	int next = 0;
	for(int i=0;i<LENGTH;i++){
		if(flex[i] == 0){
			printf("test1\n");
			len = len + 1;
			printf("len = %d\n",len);
		}
		if(flex[i] == 1 || i == LENGTH - 1 ){
			printf("test2\n");
			printf("len = %d\n",len);
			if(i == LENGTH - 1 && flex[i] == 0){
				len = len - 1;
			}
			float x = 0.5 + next;
			
			for(int j=0;j<len;j++){
				
				position[i+j-len+1][0] = x;
				
				float y = 2.0 / (len * 2) * ( (j * 2) + 1) ;
				
				position[i+j-len+1][1] = y;
			}
			len = 1;
			
			next = next + 1;
		}
	}
	for(int i=0;i<LENGTH;i++){
		printf("%f %f\n", position[i][0], position[i][1]);
	}
	
	float distance[LENGTH][LENGTH] = {0};
	
	for(int i=0;i<LENGTH;i++){
		for(int j=i+1;j<LENGTH;j++){
			float x1 = position[i][0];
			float y1 = position[i][1];
			float x2 = position[j][0];
			float y2 = position[j][1];
			if(y2 > y1){
				distance[i][j] = sqrt( (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ) ;
			}
			else{
				distance[i][j] = sqrt( (x2 - x1) * (x2 - x1) + (y1 - y2) * (y1 - y2) ) ;
			}			
		}
	}
	
	for(int i=0;i<LENGTH;i++){
		for(int j=0;j<LENGTH;j++){
			printf("%f\t", distance[i][j]);
		}
		printf("\n");
	}
	
	float totalCost[LENGTH][LENGTH] = {0};
	
	for(int i=0;i<LENGTH;i++){
		for(int j=0;j<LENGTH;j++){
			totalCost[i][j] = cost[i][j] * distance[i][j];
		}
	}
	
	for(int i=0;i<LENGTH;i++){
		for(int j=0;j<LENGTH;j++){
			printf("%f\t", distance[i][j]);
		}
		printf("\n");
	}
	
	
	return 0;
} 
