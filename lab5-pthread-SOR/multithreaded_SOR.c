/****************************************************************************


   gcc -O1 -std=gnu11 multithreaded_SOR.c -lpthread -lrt -lm -o test_mSOR

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#ifdef __APPLE__
/* Shim for Mac OS X (use at your own risk ;-) */
# include "apple_pthread_barrier.h"
#endif /* __APPLE__ */

#define CPNS 3.0    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GhZ GPU, this would be 3.2 */

#define GHOST 2   /* 2 extra rows/columns for "ghost zone". */

#define A   8   /* coefficient of x^2 */
#define B   16  /* coefficient of x */
#define C   32  /* constant term */

#define NUM_TESTS 20

/* A, B, and C needs to be a multiple of your BLOCK_SIZE,
   total array size will be (GHOST + Ax^2 + Bx + C) */

#define BLOCK_SIZE 4     // TO BE DETERMINED

#define OPTIONS 2

#define MINVAL   0.0
#define MAXVAL  10.0
//#define NUM_THREADS 4
#define TOL 0.00001
#define OMEGA 1.85       // TO BE DETERMINED

int NUM_THREADS = 4;
typedef double data_t;

typedef struct {
  long int rowlen;
  data_t *data;
} arr_rec, *arr_ptr;

// used to pass parameters to worker threads
struct thread_data
{
  int threadid;     // thread id, used for checking which section to work on
  long int len;     // how long each row is (not including ghost)
  data_t *data;     // pointer to start of assigned section
  int start;        // start indexing for row
  int end;          // end indexing for row
  int *iterations;
};
// mutex for interations
pthread_mutex_t iter_mutex = PTHREAD_MUTEX_INITIALIZER;
double global_total_change = 0.0;
pthread_mutex_t total_change_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Prototypes */
arr_ptr new_array(long int row_len);
int set_arr_rowlen(arr_ptr v, long int index);
long int get_arr_rowlen(arr_ptr v);
int init_array(arr_ptr v, long int row_len);
int init_array_rand(arr_ptr v, long int row_len);
int print_array(arr_ptr v);

void SOR(arr_ptr v, int *iterations);
void SOR_redblack(arr_ptr v, int *iterations);
void SOR_ji(arr_ptr v, int *iterations);
void SOR_blocked(arr_ptr v, int *iterations);
void SOR_multithreaded(arr_ptr v, int* iterations);
void *SOR_multithreaded_worker(void *threadarg);

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:
 
        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}
/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_REALTIME, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_REALTIME, &time_stop);
      measurement = interval(time_start, time_stop);

 */


/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/*****************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  int convergence[OPTIONS][NUM_TESTS];
  int *iterations;

  long int x, n;
  long int alloc_size;

  x = NUM_TESTS-1;
  alloc_size = GHOST + A*x*x + B*x + C;

  printf("SOR serial variations \n");

  printf("OMEGA = %0.2f\n", OMEGA);

  /* declare and initialize the array */
  arr_ptr v0 = new_array(alloc_size);

  /* Allocate space for return value */
  iterations = (int *) malloc(sizeof(int));

  OPTION = 0;
  printf("OPTION=%d (4 normal serial SOR)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR_multithreaded(v0, iterations);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = *iterations;
  }
  NUM_THREADS = 1;
  OPTION++;
  printf("OPTION=%d (1 serial SOR_redblack)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR_multithreaded(v0, iterations);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = *iterations;
  }
/*
  OPTION++;
  printf("OPTION=%d (serial SOR_ji)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR_ji(v0, iterations);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = *iterations;
  }

  OPTION++;
  printf("OPTION=%d (serial SOR_blocked)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR_blocked(v0, iterations);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = *iterations;
  }

  OPTION++;
  printf("OPTION=%d (multithreaded SOR)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR_multithreaded(v0, iterations);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = *iterations;
  }
*/
  printf("All times are in cycles (if CPNS is set correctly in code)\n");
  printf("\n");
  printf("size, SOR time, SOR iters, red/black time, red/black iters, SOR_ji time, SOR_ji iters, SOR_blocked time, SOR_blocked iters, multithreaded SOR\n");
  {
    int i, j;
    for (i = 0; i < NUM_TESTS; i++) {
      printf("%4d", A*i*i + B*i + C);
      for (OPTION = 0; OPTION < OPTIONS; OPTION++) {
        printf(", %10.4g", (double)CPNS * 1.0e9 * time_stamp[OPTION][i]);
        printf(", %4d", convergence[OPTION][i]);
      }
      printf("\n");
    }
  }

} /* end main */

/*********************************/

/* Create 2D array of specified length per dimension */
arr_ptr new_array(long int row_len)
{
  long int i;

  /* Allocate and declare header structure */
  arr_ptr result = (arr_ptr) malloc(sizeof(arr_rec));
  if (!result) {
    return NULL;  /* Couldn't allocate storage */
  }
  result->rowlen = row_len;

  /* Allocate and declare array */
  if (row_len > 0) {
    data_t *data = (data_t *) calloc(row_len*row_len, sizeof(data_t));
    if (!data) {
      free((void *) result);
      printf("\n COULDN'T ALLOCATE STORAGE \n", result->rowlen);
      return NULL;  /* Couldn't allocate storage */
    }
    result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Set row length of array */
int set_arr_rowlen(arr_ptr v, long int row_len)
{
  v->rowlen = row_len;
  return 1;
}

/* Return row length of array */
long int get_arr_rowlen(arr_ptr v)
{
  return v->rowlen;
}

/* initialize 2D array with incrementing values (0.0, 1.0, 2.0, 3.0, ...) */
int init_array(arr_ptr v, long int row_len)
{
  long int i;

  if (row_len > 0) {
    v->rowlen = row_len;
    for (i = 0; i < row_len*row_len; i++) {
      v->data[i] = (data_t)(i);
    }
    return 1;
  }
  else return 0;
}

/* initialize array with random data */
int init_array_rand(arr_ptr v, long int row_len)
{
  long int i;
  double fRand(double fMin, double fMax);

  /* Since we're comparing different algorithms (e.g. blocked, threaded
     with stripes, red/black, ...), it is more useful to have the same
     randomness for any given array size */
  srandom(row_len);
  if (row_len > 0) {
    v->rowlen = row_len;
    for (i = 0; i < row_len*row_len; i++) {
      v->data[i] = (data_t)(fRand((double)(MINVAL),(double)(MAXVAL)));
    }
    return 1;
  }
  else return 0;
}

/* print all elements of an array */
int print_array(arr_ptr v)
{
  long int i, j, row_len;

  row_len = v->rowlen;
  printf("row length = %ld\n", row_len);
  for (i = 0; i < row_len; i++) {
    for (j = 0; j < row_len; j++) {
      printf("%.4f ", (data_t)(v->data[i*row_len+j]));
    }
    printf("\n");
  }
}

data_t *get_array_start(arr_ptr v)
{
  return v->data;
}

double fRand(double fMin, double fMax)
{
  double f = (double)random() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}

/************************************/

/* SOR */
void SOR(arr_ptr v, int *iterations)
{
  long int i, j;
  long int rowlen = get_arr_rowlen(v);
  data_t *data = get_array_start(v);
  double change, total_change = 1.0e10;   /* start w/ something big */
  int iters = 0;

  while ((total_change/(double)(rowlen*rowlen)) > (double)TOL) {
    iters++;
    total_change = 0;
    for (i = 1; i < rowlen-1; i++) {
      for (j = 1; j < rowlen-1; j++) {
        change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                          data[(i+1)*rowlen+j] +
                                          data[i*rowlen+j+1] +
                                          data[i*rowlen+j-1]);
        data[i*rowlen+j] -= change * OMEGA;
        if (change < 0){
          change = -change;
        }
        total_change += change;
      }
    }
    if (abs(data[(rowlen-2)*(rowlen-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("SOR: SUSPECT DIVERGENCE iter = %ld\n", iters);
      break;
    }
  }
  *iterations = iters;
  printf("    SOR() done after %d iters\n", iters);
}

/* SOR red/black */
void SOR_redblack(arr_ptr v, int *iterations)
{
  int i, j, redblack;
  long int ti;
  long int rowlen = get_arr_rowlen(v);
  data_t *data = get_array_start(v);
  double change, total_change = 1.0e10;   /* start w/ something big */
  int iters = 0;

  ti = 0;
  redblack = 0;
  /* The while condition here tests the tolerance limit *only* when
     redblack is 0, which ensures we exit only after having done a
     full update (red + black) */
  while ((redblack == 1)
        || ((total_change/(double)(rowlen*rowlen)) > (double)TOL) )
  {
    /* Reset sum of total change only when starting a black scan. */
    if (redblack == 0) {
      total_change = 0;
    }
    for (i = 1; i < rowlen-1; i++) {
      /* The j loop needs to start at j=1 on row 0 and all even rows,
         and start at j=2 on odd rows; but when redblack is true it does
         just the opposite; and it always increments by 2. */
      for (j = 1 + ((i^redblack)&1); j < rowlen-1; j+=2) {
        change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                          data[(i+1)*rowlen+j] +
                                          data[i*rowlen+j+1] +
                                          data[i*rowlen+j-1]);
        data[i*rowlen+j] -= change * OMEGA;
        if (change < 0) {
          change = -change;
        }
        total_change += change;
        ti++;
      }
    }
    if (abs(data[(rowlen-2)*(rowlen-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("SOR: SUSPECT DIVERGENCE iter = %ld\n", iters);
      break;
    }
    redblack ^= 1;
    iters++;
  }
  /* A "red scan" only updates half of the array, and likewise for a
     "black scan"; so we need to divide iters by 2 to convert our count of
     "reds+blacks" to a count of "full scans" */
  iters /= 2;
  *iterations = iters;
  printf("    SOR_redblack() done after %d iters\n", iters);
  /* printf("ti == %ld, per iter %ld\n", ti, ti/iters); */
} /* End of SOR_redblack */

/* SOR with reversed indices */
void SOR_ji(arr_ptr v, int *iterations)
{
  long int i, j;
  long int rowlen = get_arr_rowlen(v);
  data_t *data = get_array_start(v);
  double change, total_change = 1.0e10;   /* start w/ something big */
  int iters = 0;

  while ((total_change/(double)(rowlen*rowlen)) > (double)TOL) {
    iters++;
    total_change = 0;
    for (j = 1; j < rowlen-1; j++) {
      for (i = 1; i < rowlen-1; i++) {
        change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                          data[(i+1)*rowlen+j] +
                                          data[i*rowlen+j+1] +
                                          data[i*rowlen+j-1]);
        data[i*rowlen+j] -= change * OMEGA;
        if (change < 0){
          change = -change;
        }
        total_change += change;
      }
    }
    if (abs(data[(rowlen-2)*(rowlen-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("SOR_ji: SUSPECT DIVERGENCE iter = %d\n", iters);
      break;
    }
  }
  *iterations = iters;
  printf("    SOR_ji() done after %d iters\n", iters);
}

/* SOR w/ blocking */
void SOR_blocked(arr_ptr v, int *iterations)
{
  long int i, j, ii, jj;
  long int rowlen = get_arr_rowlen(v);
  data_t *data = get_array_start(v);
  double change, total_change = 1.0e10;
  int iters = 0;
  int k;

  if ((rowlen-2) % (BLOCK_SIZE)) {
    fprintf(stderr,
"SOR_blocked: Total array size must be 2 more than a multiple of BLOCK_SIZE\n"
"(because the top/right/left/bottom rows are not scanned)\n"
"Make sure all coefficients A, B, and C are multiples of %d\n", BLOCK_SIZE);
    exit(-1);
  }

  while ((total_change/(double)(rowlen*rowlen)) > (double)TOL) {
    iters++;
    total_change = 0;
    for (ii = 1; ii < rowlen-1; ii+=BLOCK_SIZE) {
      for (jj = 1; jj < rowlen-1; jj+=BLOCK_SIZE) {
        for (i = ii; i < ii+BLOCK_SIZE; i++) {
          for (j = jj; j < jj+BLOCK_SIZE; j++) {
            change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                              data[(i+1)*rowlen+j] +
                                              data[i*rowlen+j+1] +
                                              data[i*rowlen+j-1]);
            data[i*rowlen+j] -= change * OMEGA;
            if (change < 0){
              change = -change;
            }
            total_change += change;
          }
        }
      }
    }
    if (abs(data[(rowlen-2)*(rowlen-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("SOR_blocked: SUSPECT DIVERGENCE iter = %d\n", iters);
      break;
    }
  }
  *iterations = iters;
  printf("    SOR_blocked() done after %d iters\n", iters);
} /* End of SOR_blocked */

pthread_barrier_t barrier1;
pthread_barrier_t barrier2;
pthread_barrier_t barrier3;

/* threaded worker function */
/* void *SOR_multithreaded_worker(void *threadarg)
{
  long int i, j;
  struct thread_data *my_data;
  my_data = (struct thread_data *) threadarg;
  int threadid = my_data->threadid;
  int collen = my_data->len;
  data_t *data = my_data->data;
  int start = my_data->start;
  int end = my_data->end;

  int rowlen_real = collen+GHOST;

  double change, total_change = 1.0e10;   // start w/ something big //
  int iters = 0;    // pass this in the struct

  while ((total_change/(double)(rowlen_real*rowlen_real)) > (double)TOL) {
    // barrier
    pthread_barrier_wait(&barrier1);
    if(threadid == 0)
    {
      iters++;
    }
    total_change = 0;   // barrier
    // synch so all have total_change as zero
    pthread_barrier_wait(&barrier2);
    for (i = start; i <= end; i++) {
      for (j = 1; j < collen-1; j++) {
        // have an offset for the first padded row
        change = data[i*collen+j] - .25 * (data[(i-1)*collen+j] +
                                          data[(i+1)*collen+j] +
                                          data[i*collen+j+1] +
                                          data[i*collen+j-1]);
        data[i*collen+j] -= change * OMEGA;
        if (change < 0){
          change = -change;
        }
        // barrier
        pthread_barrier_wait(&barrier3);
        total_change += change;
      }
    }
    if (abs(data[(rowlen_real-2)*(rowlen_real-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("SOR: SUSPECT DIVERGENCE iter = %ld\n", iters);
      break;
    }
  }
  pthread_mutex_lock(&iter_mutex);
  if(iters > *my_data->iterations)
  {
    *my_data->iterations = iters;
  }
  pthread_mutex_unlock(&iter_mutex);
  printf("    SOR() done after %d iters\n", iters);
}
*/

/* new version worker */
double change, total_change = 1.0e10; /* start w/ something big */
void *SOR_multithreaded_worker(void *threadarg)
{
    long int i, j;
    int rc;
    struct thread_data *my_data;
    my_data = (struct thread_data *) threadarg;
    int threadid = my_data->threadid;
    int collen = my_data->len;
    data_t *data = my_data->data;
    int start = my_data->start;
    int end = my_data->end;

    int rowlen_real = collen + GHOST;
    int iters = 0;  // Local iteration counter

    while ((total_change / (double)(rowlen_real * rowlen_real)) > (double)TOL) {
        //printf("thread %d is waiting on barrier1\n", threadid);
        rc = pthread_barrier_wait(&barrier1);
        if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
          printf("Thread %d failed on barrier wait\n", threadid);
          exit(-1);
        }
        iters++;
        total_change = 0.0; // Ensure all threads start fresh
        for (i = start; i <= end; i++) {
            for (j = 1; j < collen - 1; j++) {
                pthread_mutex_lock(&iter_mutex);
                //printf("thread %d obtained lock\n", threadid);
                change = data[i * collen + j] - 0.25 * (data[(i - 1) * collen + j] +
                                                        data[(i + 1) * collen + j] +
                                                        data[i * collen + j + 1] +
                                                        data[i * collen + j - 1]);
                data[i * collen + j] -= change * OMEGA;
                if (change < 0) {
                    change = -change;
                }
                total_change += change;
                pthread_mutex_unlock(&iter_mutex);
                //printf("thread %d relinquishes lock\n", threadid);
                //printf("thread %d is waiting on barrier2\n", threadid);
                rc = pthread_barrier_wait(&barrier2);
                if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
                  printf("Thread %d failed on barrier wait\n", threadid);
                  exit(-1);
                }
            }
        }
        if (abs(data[(rowlen_real-2)*(rowlen_real-2)]) > 10.0*(MAXVAL - MINVAL)) {
          printf("SOR: SUSPECT DIVERGENCE iter = %ld\n", iters);
          break;
        }
       // printf("thread %d is waiting on barrier3\n", threadid);
        rc = pthread_barrier_wait(&barrier3);
        if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
          printf("Thread %d failed on barrier wait\n", threadid);
          exit(-1);
        }
        //printf("thread %d has total change: %lld\n", threadid, total_change);
    }
    //printf("thread %d exited the while loop\n", threadid);

    //printf("thread %d    multi SOR() done after %d iters\n", threadid, iters);
    /* write back the iterations value */
    *(my_data->iterations) = iters;
    pthread_exit(NULL);
}
/* use pthreads to develop a calling and init function for the multithreaded worker SOR */
void SOR_multithreaded(arr_ptr v, int* iterations) {
  long int i, j, k;
  pthread_t threads[NUM_THREADS];
  struct thread_data thread_data_array[NUM_THREADS];
  int rc;
  long t;
  int len = v->rowlen - GHOST;       // number of columns to work on in a row (NUM_THREADS times more than rows)
  data_t *start_ghost = v->data;   // use this with a for loop to find which sections to work on
  int rows = len / NUM_THREADS;     // how many rows each thread will work on
  //int incrementer = rows*rowlen/NUM_THREADS;
  int offset = rows/NUM_THREADS;


  // initalize all thread_data values to pass to thread
  for(t = 0; t < NUM_THREADS; t++)
  {
    thread_data_array[t].threadid = t;    // used for computing which section to work on
    thread_data_array[t].len = len;   // number of columns to work on in a strip
    thread_data_array[t].start = 1 + t*offset;
    thread_data_array[t].end = thread_data_array[t].start + offset-1;
    thread_data_array[t].data = start_ghost;
    thread_data_array[t].iterations = iterations;
  }
  pthread_barrier_init(&barrier1, NULL, NUM_THREADS);
  pthread_barrier_init(&barrier2, NULL, NUM_THREADS);
  pthread_barrier_init(&barrier3, NULL, NUM_THREADS);

  /* dont think I need this lol 
  // getting the starting point for each thread
  for(t = 0; t < NUM_THREADS; t++)
  {
    // increment start_ghost past the padding; to the start position
    if(t == 0)
    {
      for(i = 0; i <= v.rowlen;i++)
      {
        start_ghost++;
      }
    }
    else
    {
      for(i = 0; i < incrementer; i++)
      {
        start_ghost++;
      }
    }
    thread_data_array[t].data = start_ghost;
  }
    */

  // call the helper function
  change = 1.0e10;
  total_change = 1.0e10;
  for(t = 0; t < NUM_THREADS; t++)
  {
    rc = pthread_create(&threads[t], NULL, SOR_multithreaded_worker,
      (void *) &thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }
  // JOIN THREADS HERE (Wait for all threads to finish)
  for (t = 0; t < NUM_THREADS; t++)
  {
    pthread_join(threads[t], NULL);
  }

  /* pass back the iterations value */
  iterations = thread_data_array[0].iterations;
  // destroy all barriers
  pthread_mutex_destroy(&iter_mutex);
  pthread_barrier_destroy(&barrier1);
  pthread_barrier_destroy(&barrier2);
  pthread_barrier_destroy(&barrier3);
  printf("    multi SOR() done after %d iters\n", *iterations);
}
