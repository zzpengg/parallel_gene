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
			fscanf(fPtr , "%f " , &totalCost[i][a]); /* 讀入35個字元到 facility[35] */
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

	
	FILE *OF;   /*宣告FILE資料型態的 指標*/ 
   
	OF = fopen("of.txt","w");  /* fopen function , 給予檔案名稱，和寫入方式 */
	
	for(int i=0;i<AMOUNT;i++){
		fprintf(OF, "%f\n", sumCost[i]);   /* 將字串寫入檔案 */
		fprintf(OF, "\n");
	}
	
	printf("\n%f %d", minCost[0], minCost[1]);
	
	fclose(OF); /* 關閉檔案 */
	
	return 0;
}
