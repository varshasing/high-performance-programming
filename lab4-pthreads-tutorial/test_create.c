/**************************************************************

   gcc -pthread test_create.c -o test_create -std=gnu99

 */

/* Simple thread create and exit test */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 5

/***************************************************************/
/* task 4: add sleep(3) to work before printf statement */
void *work(void *i)
{
  //sleep(3);
  printf("Hello World! from child thread %lx\n", (long)pthread_self());

  pthread_exit(NULL);
}

/****************************************************************/
int main(int argc, char *argv[])
{
  int arg,j,k,m;                  /* Local variables. */
  //pthread_t id[NUM_THREADS];
  /* task 3, id is declared using malloc
   * pass the location to store the thread ID information
   * in pointer form instead of a referenced array
   * task 5: sleep(3) before returning in main
   */
  pthread_t* id  = (pthread_t*)malloc(NUM_THREADS * (sizeof(pthread_t)));

  printf("Hello test_create.c\n");

  for (long t = 0; t < NUM_THREADS; ++t) {
    if (pthread_create(id++, NULL, work, NULL)) {
      printf("ERROR creating the thread\n");
      exit(19);
    }
  }

  printf("main() after creating the thread.  My id is %lx\n",
                                              (long) pthread_self());
  sleep(3);
  return(0);
} /* end main */
