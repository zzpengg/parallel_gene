#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LENGTH 35
#define AMOUNT 500

int test(int ch[35], short int flex[34]);

int main(){
	FILE *fPtr;
	
	fPtr=fopen("random.txt","r");
	int facility[AMOUNT][LENGTH];
	short int flex[AMOUNT][LENGTH-1] ;
	for(int i=0;i<AMOUNT;i++){
		for(int a=0;a<LENGTH;a++){
			fscanf(fPtr , "%d " , &facility[i][a]); /* Ū�J35�Ӧr���� facility[35] */
		}
	
		for(int a=0;a<LENGTH-1;a++){
			fscanf(fPtr , "%hd " , &flex[i][a]); /* Ū�J34�Ӧr���� flex[34] */
		}
		
	}
	
	for(int a=0;a<AMOUNT;a++){
		test(facility[a], flex[a]);
	}
	
	fclose(fPtr);
	
	return 0;
}

int test(int ch[35], short int flex[34]){
	float cost[LENGTH][LENGTH];
	for(int i=0;i<LENGTH;i++){
		for(int j=0;j<LENGTH;j++){
			cost[i][j] = 1.0f;
		}
	}
	float position[LENGTH][2];
//	printf("cost = %f\n",cost[0][0]);
	
	int H = 2;
	int W = 1;
	
	int len = 1;
	int next = 0;
	for(int i=0;i<LENGTH;i++){
		if(flex[i] == 0){
//			printf("test1\n");
			len = len + 1;
//			printf("len = %d\n",len);
		}
		if(flex[i] == 1 || i == LENGTH - 1 ){
//			printf("test2\n");
//			printf("len = %d\n",len);
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
	
	float totalCost[LENGTH][LENGTH] = {0};
	
	for(int i=0;i<LENGTH;i++){
		for(int j=0;j<LENGTH;j++){
			totalCost[i][j] = cost[i][j] * distance[i][j];
		}
	}
	
	FILE *dis;   /*�ŧiFILE��ƫ��A�� ����*/ 
   
	dis = fopen("distance.txt","a");  /* fopen function , �����ɮצW�١A�M�g�J�覡 */
	
	for(int i=0;i<LENGTH;i++){
		for(int j=0;j<LENGTH;j++){
			fprintf(dis, "%f\t", distance[i][j]);   /* �N�r��g�J�ɮ� */
		}
		fprintf(dis, "\n");
	}
	
	fclose(dis); /* �����ɮ� */
	
	FILE *cos;   /*�ŧiFILE��ƫ��A�� ����*/ 
   
	cos = fopen("totalCost.txt","a");  /* fopen function , �����ɮצW�١A�M�g�J�覡 */
	
	for(int i=0;i<LENGTH;i++){
		for(int j=0;j<LENGTH;j++){
			fprintf(cos, "%f\t", totalCost[i][j]);   /* �N�r��g�J�ɮ� */
		}
		fprintf(cos, "\n");
	}
	
	fclose(cos); /* �����ɮ� */
	
	return 0;
}
