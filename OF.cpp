#include <stdio.h>
#include <stdlib.h>

#define LENGTH 35
#define AMOUNT 5
int main(){
	printf("test1");
	FILE *fPtr;
	
	fPtr=fopen("totalCost.txt","r");
	printf("test");
	float totalCost[AMOUNT][LENGTH*LENGTH];
	for(int i=0;i<AMOUNT;i++){
		for(int a=0;a<LENGTH*LENGTH;a++){
			fscanf(fPtr , "%f " , &totalCost[i][a]); /* Ū�J35�Ӧr���� facility[35] */
		}
	}
	
	fclose(fPtr);
	float sumCost[AMOUNT] ;
	float minCost[2];
	for(int i=0;i<AMOUNT;i++){
		for(int a=0;a<LENGTH*LENGTH;a++){
			sumCost[i] += totalCost[i][a];
		}
		if(i==0){
			minCost[0] = sumCost[0];
			minCost[1] = 0;
		}else if(minCost[0] > sumCost[i]){
			minCost[0] = sumCost[i];
			minCost[1] = i;
		}
	}

	
	FILE *OF;   /*�ŧiFILE��ƫ��A�� ����*/ 
   
	OF = fopen("of.txt","w");  /* fopen function , �����ɮצW�١A�M�g�J�覡 */
	
	for(int i=0;i<AMOUNT;i++){
		fprintf(OF, "%f\n", sumCost[i]);   /* �N�r��g�J�ɮ� */
		fprintf(OF, "\n");
	}
	
	printf("\n%f %d", minCost[0], minCost[1]);
	
	fclose(OF); /* �����ɮ� */
	
	return 0;
}
