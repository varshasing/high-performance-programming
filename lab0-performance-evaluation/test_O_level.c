/**********************************************************************

 gcc -O0 test_O_level.c -o test_O_level
 gcc -O1 test_O_level.c -o test_O_level

 gcc -O0 -S test_O_level.c
 gcc -O1 -S test_O_level.c

*/

#include <stdio.h>

/*********************************************************************/
int main(int argc, char *argv[])
{
  long long int i, j, k, steps = 0;
  double quasi_random = 0;

  printf("\n Starting a loop \n");
  
  for (i = 0; i <= 200000000; i++) {
    quasi_random = quasi_random*quasi_random - 1.923432;
  }
  
  printf("The variable calculated is %11.9f\n", quasi_random);
  printf("\n done \n");
}
