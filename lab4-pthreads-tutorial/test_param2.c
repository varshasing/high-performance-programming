/*************************************************************************

  gcc -pthread test_param2.c -o test_param2 -std=gnu99

 */

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#define NUM_THREADS 10

int unique = 0;
/********************/
void *work(void *i)
{
  /* task 9: change the number of threads executed */
  int *t = (int *)i;
  *t += NUM_THREADS;
  long int k;
  int f = *((int*)(i));  // get the value being pointed to
  int *g = (int*)(i);    // get the pointer itself
  int *n = g + unique;
  unique++;
  *n = *n+5;
  /* busy work */
  k = 0;
  for (long j=0; j < 10000000; j++) {
    k += j;
  }
  f -= 1;
   *g -= 5;
  /* printf("Hello World from %lu with value %d\n", pthread_self(), f); */
  
  printf("in work(): f=%2d, k=%ld, *g=%d\n", f, k, *g);

  pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
  long k, t;
  pthread_t id[NUM_THREADS];
  int array[NUM_THREADS];
  for (t = 0; t < NUM_THREADS; ++t)
  {
    array[t] = t;
  }
  printf("The array before:\n");
  for (t = 0; t < NUM_THREADS; ++t)
  {
    printf("%d ",array[t]); 
  }

  for (t = 0; t < NUM_THREADS; ++t) {
    if (pthread_create(&id[t], NULL, work, (void *)(array))) {
      printf("ERROR creating the thread\n");
      exit(19);
    }
   pthread_join(id[t], NULL);
  }

  printf("\n the array after\n");
  for (t = 0; t < NUM_THREADS; ++t)
  {
    printf("%d ",array[t]);
  }

  /* busy work */
  k = 0;
  for (long j=0; j < 100000000; j++) {
    k += j;
  }

  printf("\nk=%ld\n", k);
  printf("After creating the threads.  My id is %lx, t = %d\n",
                                                 (long)pthread_self(), t);

  return(0);

} /* end main */
