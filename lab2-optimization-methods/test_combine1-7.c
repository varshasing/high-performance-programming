/****************************************************************************

 gcc -O1 test_combine1-7.c -lrt -o test_combine

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

/* We want to test a range of work sizes. We will generate these
   using the quadratic formula:  A x^2 + B x + C                     */
#define A   1   /* coefficient of x^2 */
#define B   6   /* coefficient of x */
#define C   10  /* constant term */

#define NUM_TESTS 15   /* Number of different sizes to test */


#define OUTER_LOOPS 2000

#define CPNS 3.0    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GHz GPU, this would be 3.2 */

#define OPTIONS 7

/* Type of operation. This can be multiplication or addition.
   for addition, IDENT should be 0.0 and OP should be +
   for multiplication, IDENT should be 1.0 and OP should be *
 */
#define IDENT 0.0
#define OP +

/* Type of data being "combined". This can be any of the types:
   int, long int, float, double, long double */
typedef float data_t;

/* Create abstract data type for an array in memory */
typedef struct {
  long int len;
  data_t *data;
} array_rec, *array_ptr;

/* Prototypes */
array_ptr new_array(long int len);
int get_array_element(array_ptr v, long int index, data_t *dest);
long int get_array_length(array_ptr v);
int set_array_length(array_ptr v, long int index);
int init_array(array_ptr v, long int len);
void combine1(array_ptr v, data_t *dest);
void combine2(array_ptr v, data_t *dest);
void combine3(array_ptr v, data_t *dest);
void combine4(array_ptr v, data_t *dest);
void combine5_2(array_ptr v, data_t *dest);
void combine6(array_ptr v, data_t *dest);
void combine7(array_ptr v, data_t *dest);
void combine5_3(array_ptr v, data_t *dest);
void combine5_5(array_ptr v, data_t *dest);
void combine5_7(array_ptr v, data_t *dest);
void combine5_9(array_ptr v, data_t *dest);
void combine5_10(array_ptr v, data_t *dest);
void combine6_1(array_ptr v, data_t *dest);
void combine7_1(array_ptr v, data_t *dest);
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
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      measurement = interval(time_start, time_stop);

 */


/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i=1; i<j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}


/*****************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  double final_answer = 0;
  long int x, n, k, alloc_size;
  data_t *result;

  printf("Vector reduction (combine) examples\n");

  wakeup_delay();

  final_answer = wakeup_delay();

  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;
  long vector_size = x*alloc_size*sizeof(data_t);
  printf("size of vector: %ld\n", vector_size);
  /* check that the largest size of vector fits in level 1 cache (32K bytes) */
  if(vector_size > sysconf(_SC_LEVEL1_DCACHE_SIZE))
  {
 	perror("vector will not fit in L1 cache\n");
	return 1;
  }
  printf("vector will fit in L1 cache: %ld bytes (%.2f KiB)\n\n", vector_size, vector_size/1024.0);
  
  /* declare and initialize the arrays */
  array_ptr v0 = new_array(alloc_size);
  init_array(v0, alloc_size);
  result = (data_t *) malloc(sizeof(data_t));

  printf("Testing %d variants of combine(),\n", OPTIONS);
  printf("  on arrays of %d sizes from %d to %ld\n", NUM_TESTS, C, alloc_size);

  /* execute and time all 7 options from B&O  */
  OPTION = 0;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for(k=0; k<OUTER_LOOPS; k++) {
      combine6_1(v0, result);
      final_answer += *result;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for(k=0; k<OUTER_LOOPS; k++) {
      combine7_1(v0, result);
      final_answer += *result;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for(k=0; k<OUTER_LOOPS; k++) {
      //combine5_3(v0, result);
      final_answer += *result;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for(k=0; k<OUTER_LOOPS; k++) {
      //combine5_5(v0, result);
      final_answer += *result;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (k=0; k<OUTER_LOOPS; k++) {
      //combine5_7(v0, result);
      final_answer += *result;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for(k=0; k<OUTER_LOOPS; k++) {
      //combine5_9(v0, result);
      final_answer += *result;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for(k=0; k<OUTER_LOOPS; k++) {
      //combine5_10(v0, result);
      final_answer += *result;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }
  
  /* output times */
  printf("%s", "size, combine6_1, combine7_1, combine5_3, combine5_5, combine5_7, combine5_9, combine5_10\n");
  {
    int i, j;
    for (i = 0; i < NUM_TESTS; i++) {
      printf("%d,  ", (A*i*i + B*i + C) * OUTER_LOOPS );
      for (j = 0; j < OPTIONS; j++) {
        if (j != 0) {
          printf(", ");
        }
        printf("%ld", (long int)((double)(CPNS) * 1.0e9 * time_stamp[j][i]));
      }
      printf("\n");
    }
  }

  printf("\n");
  printf("Sum of all results: %g\n", final_answer);

  return 0;  
} /* end main */

/**********************************************/
/* Create array of specified length */
array_ptr new_array(long int len)
{
  long int i;

  /* Allocate and declare header structure */
  array_ptr result = (array_ptr) malloc(sizeof(array_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = len;

  /* Allocate and declare array */
  if (len > 0) {
    data_t *data = (data_t *) calloc(len, sizeof(data_t));
    if (!data) {
      free((void *) result);
      return NULL;  /* Couldn't allocate storage */
    }
    result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Retrieve array element and store at dest.
   Return 0 (out of bounds) or 1 (successful)
*/
int get_array_element(array_ptr v, long int index, data_t *dest)
{
  if (index < 0 || index >= v->len) {
    return 0;
  }
  *dest = v->data[index];
  return 1;
}

/* Return length of array */
long int get_array_length(array_ptr v)
{
  return v->len;
}

/* Set length of array */
int set_array_length(array_ptr v, long int index)
{
  v->len = index;
  return 1;
}

/* initialize an array */
int init_array(array_ptr v, long int len)
{
  long int i;

  if (len > 0) {
    v->len = len;
    for (i = 0; i < len; i++) {
      v->data[i] = (data_t)(i+1);
    }
    return 1;
  }
  else return 0;
}

data_t *get_array_start(array_ptr v)
{
  return v->data;
}


/*************************************************/
/* Combine1:  Implementation with maximum use of data abstraction */
void combine1(array_ptr v, data_t *dest)
{
  long int i;
  long int get_array_length(array_ptr v);

  *dest = IDENT;
  for (i = 0; i < get_array_length(v); i++) {
    data_t val;
    get_array_element(v, i, &val);
    *dest = *dest OP val;
  }
}

/* Combine2:  Move call to array_length out of loop 
 * Example of --> Code motion */
void combine2(array_ptr v, data_t *dest)
{
  long int i;
  long int get_array_length(array_ptr v);
  long int length = get_array_length(v);

  *dest = IDENT;
  for (i = 0; i < length; i++) {
    data_t val;
    get_array_element(v, i, &val);
    *dest = *dest OP val;
  }
}

/* Combine3:  Direct access to array
 * Example of --> Reduce procedure calls */
void combine3(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  data_t *data = get_array_start(v);

  *dest = IDENT;
  for (i = 0; i < length; i++) {
    *dest = *dest OP data[i];
  }
}

/* Combine4:  Accumulate result in local variable
 * Example of --> Eliminate unneeded memory references */
void combine4(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  data_t *data = get_array_start(v);
  data_t acc = IDENT;

  for (i = 0; i < length; i++) {
    acc = acc OP data[i];
  }
  *dest = acc;
}

/* Combine5:  Unroll loop by 2
 * Example of --> Loop unrolling */
void combine5(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 1;
  data_t *data = get_array_start(v);
  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=2) {
    acc = (acc OP data[i]) OP data[i+1];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc = acc OP data[i];
  }
  *dest = acc;
}

/* * * * * COMBINE 6 AND COMBINE 6 IMPLEMENTATION * * * * * * * * * */
/* Combine5:  Unroll loop by 2
 * Example of --> Loop unrolling */
void combine56_2(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 1;
  data_t *data = get_array_start(v);
  data_t acc0 = IDENT;
  data_t acc1 = IDENT;
  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=2) 
  {
    acc0 = acc0 OP data[i];
    acc1 = acc1 OP data[i+1];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc0 = acc0 OP data[i];
  }
  *dest = acc0 OP acc1;
}
/* Combine5:  Unroll loop by 3
 * Example of --> Loop unrolling */
void combine56_3(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 2;
  data_t *data = get_array_start(v);
  data_t acc0 = IDENT;
  data_t acc1 = IDENT;
  data_t acc2 = IDENT;

  /* Combine three elements at a time */
  for (i = 0; i < limit; i+=3) {
    acc0 = acc0 OP data[i];
    acc1 = acc1 OP data[i+1];
    acc2 = acc2 OP data[i+2];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc0 = acc0 OP data[i];
  }
  *dest = acc0 OP acc1 OP acc2;
}

/* Combine5:  Unroll loop by 5
 * Example of --> Loop unrolling */
void combine56_5(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 4;
  data_t *data = get_array_start(v);
  data_t acc0 = IDENT;
  data_t acc1 = IDENT;
  data_t acc2 = IDENT;
  data_t acc3 = IDENT;
  data_t acc4 = IDENT;

  /* Combine five elements at a time */
  for (i = 0; i < limit; i+=5) {
    acc0 = acc0 OP data[i];
    acc1 = acc1 OP data[i+1];
    acc2 = acc2 OP data[i+2];
    acc3 = acc3 OP data[i+3];
    acc4 = acc4 OP data[i+4];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc0 = acc0 OP data[i];
  }
  *dest = acc0 OP acc1 OP acc2 OP acc3 OP acc4;
}

/* Combine5:  Unroll loop by 7
 * Example of --> Loop unrolling */
void combine56_7(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 6;
  data_t *data = get_array_start(v);
  data_t acc0 = IDENT;
  data_t acc1 = IDENT;
  data_t acc2 = IDENT;
  data_t acc3 = IDENT;
  data_t acc4 = IDENT;
  data_t acc5 = IDENT;
  data_t acc6 = IDENT;

  /* Combine five elements at a time */
  for (i = 0; i < limit; i+=7) {
    acc0 = acc0 OP data[i];
    acc1 = acc1 OP data[i+1];
    acc2 = acc2 OP data[i+2];
    acc3 = acc3 OP data[i+3];
    acc4 = acc4 OP data[i+4];
    acc5 = acc5 OP data[i+5];
    acc6 = acc6 OP data[i+6];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc0 = acc0 OP data[i];
  }
  *dest = acc0 OP acc1 OP acc2 OP acc3 OP acc4 OP acc5 OP acc6;
}

/* Combine5:  Unroll loop by 9
 * Example of --> Loop unrolling */
void combine56_9(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 8;
  data_t *data = get_array_start(v);
  data_t acc0 = IDENT;
  data_t acc1 = IDENT;
  data_t acc2 = IDENT;
  data_t acc3 = IDENT;
  data_t acc4 = IDENT;
  data_t acc5 = IDENT;
  data_t acc6 = IDENT;
  data_t acc7 = IDENT;
  data_t acc8 = IDENT;

  /* Combine five elements at a time */
  for (i = 0; i < limit; i+=9) {
    acc0 = acc0 OP data[i];
    acc1 = acc1 OP data[i+1];
    acc2 = acc2 OP data[i+2];
    acc3 = acc3 OP data[i+3];
    acc4 = acc4 OP data[i+4];
    acc5 = acc5 OP data[i+5];
    acc6 = acc6 OP data[i+6];
    acc7 = acc7 OP data[i+7];
    acc8 = acc8 OP data[i+8];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc0 = acc0 OP data[i];
  }
  *dest = acc0 OP acc1 OP acc2 OP acc3 OP acc4 OP acc5 OP acc6 OP acc7 OP acc8;
}

/* Combine5:  Unroll loop by 10
 * Example of --> Loop unrolling */
void combine56_10(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 9;
  data_t *data = get_array_start(v);
  data_t acc0 = IDENT;
  data_t acc1 = IDENT;
  data_t acc2 = IDENT;
  data_t acc3 = IDENT;
  data_t acc4 = IDENT;
  data_t acc5 = IDENT;
  data_t acc6 = IDENT;
  data_t acc7 = IDENT;
  data_t acc8 = IDENT;
  data_t acc9 = IDENT;

  /* Combine five elements at a time */
  for (i = 0; i < limit; i+=10) {
    acc0 = acc0 OP data[i];
    acc1 = acc1 OP data[i+1];
    acc2 = acc2 OP data[i+2];
    acc3 = acc3 OP data[i+3];
    acc4 = acc4 OP data[i+4];
    acc5 = acc5 OP data[i+5];
    acc6 = acc6 OP data[i+6];
    acc7 = acc7 OP data[i+7];
    acc8 = acc8 OP data[i+8];
    acc9 = acc9 OP data[i+9];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc0 = acc0 OP data[i];
  }
  *dest = acc0 OP acc1 OP acc2 OP acc3 OP acc4 OP acc5 OP acc6 OP acc7 OP acc8 OP acc9;
}

/* * * * * * * COMBINE 7 AND COMBINE 5 IMPLEMENTATION * * * * * * */

/* Combine5:  Unroll loop by 2
 * Example of --> Loop unrolling */
void combine57_2(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 1;
  data_t *data = get_array_start(v);
  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=2) {
    acc = acc OP (data[i] OP data[i+1]);
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc = acc OP data[i];
  }
  *dest = acc;
}

/* Combine5:  Unroll loop by 3
 * Example of --> Loop unrolling */
void combine57_3(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 2;
  data_t *data = get_array_start(v);
  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=3) {
    acc = acc OP ((data[i] OP data[i+1]) OP data[i+2]);
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc = acc OP data[i];
  }
  *dest = acc;
}

/* Combine5:  Unroll loop by 5
 * Example of --> Loop unrolling */
void combine57_5(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 4;
  data_t *data = get_array_start(v);
  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=5) {
    acc = acc OP ((((data[i] OP data[i+1]) OP data[i+2]) OP data[i+3]) OP data[i+4]);
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc = acc OP data[i];
  }
  *dest = acc;
}

/* Combine5:  Unroll loop by 7
 * Example of --> Loop unrolling */
void combine57_7(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 6;
  data_t *data = get_array_start(v);
  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=7) {
    acc = acc OP ((((((data[i] OP data[i+1]) OP data[i+2]) OP data[i+3]) OP data[i+4]) OP data[i+5]) OP data[i+6]);
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc = acc OP data[i];
  }
  *dest = acc;
}

/* Combine5:  Unroll loop by 9
 * Example of --> Loop unrolling */
void combine57_9(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 8;
  data_t *data = get_array_start(v);
  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=9) {
    acc = acc OP ((((((((data[i] OP data[i+1]) OP data[i+2]) OP data[i+3]) OP data[i+4]) OP data[i+5]) OP data[i+6]) OP data[i+7]) OP data[i+8]);
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc = acc OP data[i];
  }
  *dest = acc;
}

/* Combine5:  Unroll loop by 10
 * Example of --> Loop unrolling */
void combine57_10(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 9;
  data_t *data = get_array_start(v);
  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=10) {
    acc = acc OP (((((((((data[i] OP data[i+1]) OP data[i+2]) OP data[i+3]) OP data[i+4]) OP data[i+5]) OP data[i+6]) OP data[i+7]) OP data[i+8]) OP data[i+9]);
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc = acc OP data[i];
  }
  *dest = acc;
}
// general version for combine6

void combine6_1(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length;
  data_t *data = get_array_start(v);
  data_t acc0 = IDENT;

/* combine one element at a time w/ 1 acccumulator */
  for(i = 0; i < limit; i+=1)
    {
      acc0 = acc0 OP data[i];
    }
  
  /* finish remaining elements */
  for(; i < length; i++)
  {
    acc0 = acc0 OP data[i];
  }
  *dest = acc0;
}
/* Combine6:  Unroll loop by 2, 2 accumulators
 * Example of --> parallelism */
void combine6(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 1;
  data_t *data = get_array_start(v);
  data_t acc0 = IDENT;
  data_t acc1 = IDENT;

  /* Combine two elements at a time w/ 2 accumulators */
  for (i = 0; i < limit; i+=2) {
    acc0 = acc0 OP data[i];
    acc1 = acc1 OP data[i+1];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc0 = acc0 OP data[i];
  }
  *dest = acc0 OP acc1;
}

/* Combine7:  Unroll loop by 2, change associativity
 * Example of --> parallelism */
void combine7_1(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length;
  data_t *data = get_array_start(v);
  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=1) {
    acc = (acc OP data[i]);
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc = acc OP data[i];
  }
  *dest = acc;
}

/* Combine7:  Unroll loop by 2, change associativity
 * Example of --> parallelism */
void combine7(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 1;
  data_t *data = get_array_start(v);
  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=2) {
    acc = acc OP (data[i] OP data[i+1]);
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc = acc OP data[i];
  }
  *dest = acc;
}
