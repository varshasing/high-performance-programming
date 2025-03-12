/**********************************
Data import/export demo code
 by Daniel Collins (2011S) and Robert Munafo (2021S)
**********************************/

#include <stdio.h>
#include <stdlib.h>

/* Open-Office-Compatible printout */
void print_ooc()
{
  long int i, j, k, p10;

  printf("\n");
  printf("Table of data without an X-axis column:\n");
  for (j=0;j<9;j++) {
    printf("col%ld",j);
    j==8 ? printf(" ") : printf(",");
  }
  for (i=0;i<10;i++) {
    for (j=0;j<i;j++) {
      if (j > 0) { printf(", "); }
      printf("%3ld", i*j);
    }
    printf("\n");
  }
  printf("\n");

  printf("\n");
  printf("The same table of data with an X-axis column added:\n");
  printf(" x ");
  for(j=0;j<9;j++) {
    printf(",col%ld",j);
  }
  printf("\n");
  for(i=1; i<10; i++) {
    printf("%3ld", i*10);
    for (j=0;j<i;j++) {
      printf(", %3ld", i*j);
    }
    printf("\n");
  }
  printf("\n");

  printf("\n");
  printf("The same table of data with powers of 10 in the X-axis column:\n");
  printf("      x ");
  for (j=0; j<9; j++) {
    printf(", col%ld", j);
  }
  printf("\n");
  p10 = 1;
  for (i=1; i<10; i++) {
    p10 *= 10;
    printf("%8ld", p10);
    for (j=0; j<i; j++) {
      printf(", %3ld", i*j);
    }
    printf("\n");
  }
  printf("\n");
}

/* MATLAB-compatible printout */
void print_matlab()
{
  int i, j;

  printf("\n");
  printf("Triangular data filled with NaN to make it rectangular, more\n");
  printf("suitable for importing into MATLAB:\n");
  for (i=0; i<10; i++) {
    /* First i cells in the row should be multiples of i */
    for (j=0; j<i; j++) {
      if (j > 0) { printf(", "); }
      printf("%3d", i*j);
    }
    /* MATLAB needs all rows to have data in all cells -- but our data is not
       like that. We can fill in the blank cells with NaN (not a number) to
       fulfill the requirement. */
    for (j=i; j<10; j++) {
      if (j > 0) { printf(", "); }
      printf("NaN");
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char *argv[])
{
  print_ooc();
  print_matlab();
  printf("\n");
  return 0;
}
