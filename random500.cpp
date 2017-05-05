# include <stdio.h>
# include <stdlib.h>
# include <time.h>

#define N 35

void shuffle(int*);

int main(void) {
    srand(time(NULL));
    
//    Card cards[N];
//    shuffle(cards);
//    int i;
//    for(i = 0; i < N; i++) {
//        printf("%s%s%c", cards[i].suit, cards[i].symbol, 
//            (i + 1) % 13 ? ' ' : '\n');
//    }
    
    int facility[35];
    for(int i=0;i<35;i++){
    	facility[i] = i;
	}

	FILE *fPtr;   /*�ŧiFILE��ƫ��A�� ����*/ 
   
    fPtr = fopen("random.txt","w");  /* fopen function , �����ɮצW�١A�M�g�J�覡 */
	for(int k=0;k<500;k++){
		shuffle(facility);
		for(int i=0;i<35;i++){
	    	fprintf(fPtr, "%i ", facility[i] );   /* �N�r��g�J�ɮ� */
		}
		fprintf(fPtr, "\n");   /* �N�r��g�J�ɮ� */
		for(int i=0;i<34;i++){		
			int j = rand() % 2;
	    	fprintf(fPtr, "%i ",j);   /* �N�r��g�J�ɮ� */
		}
		fprintf(fPtr, "\n");   /* �N�r��g�J�ɮ� */
	}
	
	fclose(fPtr); /* �����ɮ� */ 

    return 0;
} 


void shuffle(int* facility) {
    int i;
    for(i = 0; i < N; i++) {
        int j = rand() % 35;
        int tmp = facility[i];
        facility[i] = facility[j];
        facility[j] = tmp;
    }
}
