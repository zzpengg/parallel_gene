#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define FACILITY 20 

int main(){

  srand(time(NULL));

  FILE *fPtr;

  fPtr=fopen("cost.txt","w");

  for(int i=1;i<=FACILITY;i++){
    for(int j=i+1;j<=FACILITY;j++){
      int r = rand() % 6;
      while(r == 0){
        r = rand() % 6;
      }
      printf("%d %d %d\n", i, j, r);
      fprintf(fPtr, "%d %d %d\n", i, j, r);
    }
  }

  fclose(fPtr);
  return 0;
}
