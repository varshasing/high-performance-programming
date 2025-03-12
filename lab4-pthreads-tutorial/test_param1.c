/*************************************************************************

  gcc -pthread test_param1.c -o test_param -std=gnu99

 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

/************************************************************************/
/* task 7: try printing threadid and passing it a signed char, see the difference */
void *PrintHello(void *threadid)
{

  printf("Before switching, PrintHello() in thread # %ld ! \n", threadid);
  long tid;

  tid = (long) threadid;

  printf("PrintHello() in thread # %ld ! \n", tid);

  pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
  pthread_t threads[NUM_THREADS];
  int rc;
  signed char c;

  printf("Hello test_param1.c\n");

  for (long t = 0; t < NUM_THREADS; t++) {
    c = -12;
    printf("In main:  creating thread %d\n", t);
    //rc = pthread_create(&threads[t], NULL, PrintHello, (void*) t);
    rc = pthread_create(&threads[t], NULL, PrintHello, (void*) (c));
    if (rc) 
    {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
    pthread_join(threads[t], NULL);
  }
  printf("It's me MAIN -- Good Bye World!\n");

  pthread_exit(NULL);

} /* end main */
