/*****************************************************************************/

/*
  gcc -O1 -fopenmp MMM.c -lrt -lm -o test_mmm
  OMP_NUM_THREADS=2 ./test_mmm
*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define CPNS 3.0    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GhZ GPU, this would be 3.2 */

/* We want to test a wide range of work sizes. We will generate these
   using the quadratic formula:  A x^2 + B x + C                     */
#define A   0  /* coefficient of x^2 */
#define B   64  /* coefficient of x */
#define C   64  /* constant term */

#define NUM_TESTS 64  /* Number of different sizes to test */

#define OPTIONS 1
#define IDENT 0
#define BLOCKS 32 /* number of blocks in the matrix */

typedef double data_t;

/* Create abstract data type for matrix */
typedef struct {
  long int len;
  data_t *data;
} matrix_rec, *matrix_ptr;

/* Prototypes */
int clock_gettime(clockid_t clk_id, struct timespec *tp);
matrix_ptr new_matrix(long int row_len);
int set_matrix_row_length(matrix_ptr m, long int row_len);
long int get_matrix_row_length(matrix_ptr m);
int init_matrix(matrix_ptr m, long int row_len);
int zero_matrix(matrix_ptr m, long int row_len);
void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mmm_jki(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void bijk(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void bkij(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void bjki(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void bkij_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c);

/* This define is only used if you do not set the environment variable
   OMP_NUM_THREADS as instructed above, and if OpenMP also does not
   automatically detect the hardware capabilities.

   If you have a machine with lots of cores, you may wish to test with
   more threads, but make sure you also include results for THREADS=4
   in your report. */
#define THREADS 4

void detect_threads_setting()
{
  long int i, ognt;
  char * env_ONT;

  /* Find out how many threads OpenMP thinks it is wants to use */
#pragma omp parallel for
  for(i=0; i<1; i++) {
    ognt = omp_get_num_threads();
  }

  printf("omp's default number of threads is %d\n", ognt);

  /* If this is illegal (0 or less), default to the "#define THREADS"
     value that is defined above */
  if (ognt <= 0) {
    if (THREADS != ognt) {
      printf("Overriding with #define THREADS value %d\n", THREADS);
      ognt = THREADS;
    }
  }

  omp_set_num_threads(ognt);

  /* Once again ask OpenMP how many threads it is going to use */
#pragma omp parallel for
  for(i=0; i<1; i++) {
    ognt = omp_get_num_threads();
  }
  printf("Using %d threads for OpenMP\n", ognt);
}


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
  double wakeup_answer;
  long int x, n, alloc_size;

  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;

  printf("Dense MMM tests \n\n");

  wakeup_answer = wakeup_delay();

  printf("Doing MMM three different ways,\n");
  printf("for %d different matrix sizes from %d to %d\n",
                                                     NUM_TESTS, C, alloc_size);
  printf("This may take a while!\n\n");

  /* declare and initialize the matrix structure */
  matrix_ptr a0 = new_matrix(alloc_size);
  init_matrix(a0, alloc_size);
  matrix_ptr b0 = new_matrix(alloc_size);
  init_matrix(b0, alloc_size);
  matrix_ptr c0 = new_matrix(alloc_size);
  zero_matrix(c0, alloc_size);

  OPTION = 0;
/*  
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    set_matrix_row_length(a0, n);
    set_matrix_row_length(b0, n);
    set_matrix_row_length(c0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    mmm_ijk(a0, b0, c0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }
  OPTION++;
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    set_matrix_row_length(a0, n);
    set_matrix_row_length(b0, n);
    set_matrix_row_length(c0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    mmm_kij(a0, b0, c0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    set_matrix_row_length(a0, n);
    set_matrix_row_length(b0, n);
    set_matrix_row_length(c0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    bkij(a0, b0, c0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }
*/
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    set_matrix_row_length(a0, n);
    set_matrix_row_length(b0, n);
    set_matrix_row_length(c0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    bkij_omp(a0, b0, c0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

/*
  OPTION++;
  if (OPTIONS > 2) {
    for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
      printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
      set_matrix_row_length(a0, n);
      set_matrix_row_length(b0, n);
      set_matrix_row_length(c0, n);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      mmm_jki(a0, b0, c0);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      time_stamp[OPTION][x] = interval(time_start, time_stop);
    }
  }

  
  printf("BLOCKING\n");
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    set_matrix_row_length(a0, n);
    set_matrix_row_length(b0, n);
    set_matrix_row_length(c0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    bijk(a0, b0, c0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }
 
  printf("NONBLOCKING\n");
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    set_matrix_row_length(a0, n);
    set_matrix_row_length(b0, n);
    set_matrix_row_length(c0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    mmm_ijk(a0, b0, c0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }


  OPTION++;
  if (OPTIONS > 2) {
    for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
      printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
      set_matrix_row_length(a0, n);
      set_matrix_row_length(b0, n);
      set_matrix_row_length(c0, n);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      bjki(a0, b0, c0);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      time_stamp[OPTION][x] = interval(time_start, time_stop);
    }
  }
*/

  printf("Done collecting measurements.\n\n");

  printf("row_len, bkij_omp\n");
  {
    int i, j;
    for (i = 0; i < NUM_TESTS; i++) {
      printf("%ld, ", A*i*i + B*i + C);
      for (j = 0; j < OPTIONS; j++) {
        if (j != 0) {
          printf(", ");
        }
        printf("%ld", (long int) ((double)(CPNS) * 1.0e9 * time_stamp[j][i]));
      }
      printf("\n");
    }
  }
  printf("\n");

  printf("Wakeup delay computed: %g \n", wakeup_answer);
} /* end main */


/**********************************************/

/* Create matrix of specified length */
matrix_ptr new_matrix(long int row_len)
{
  long int i;
  long int alloc;

  /* Allocate and declare header structure */
  matrix_ptr result = (matrix_ptr) malloc(sizeof(matrix_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = row_len;

  /* Allocate and declare array */
  if (row_len > 0) {
    alloc = row_len * row_len;
    data_t *data = (data_t *) calloc(alloc, sizeof(data_t));
    if (!data) {
          free((void *) result);
          printf("\n COULDN'T ALLOCATE %ld BYTES STORAGE \n",
                                                       alloc * sizeof(data_t));
          return NULL;  /* Couldn't allocate storage */
        }
        result->data = data;
  } else {
    result->data = NULL;
  }

  return result;
}

/* Set length of matrix */
int set_matrix_row_length(matrix_ptr m, long int row_len)
{
  m->len = row_len;
  return 1;
}

/* Return length of matrix */
long int get_matrix_row_length(matrix_ptr m)
{
  return m->len;
}

/* initialize matrix */
int init_matrix(matrix_ptr m, long int row_len)
{
  long int i;

  if (row_len > 0) {
    m->len = row_len;
    for (i = 0; i < row_len*row_len; i++) {
      m->data[i] = (data_t)(i);
    }
    return 1;
  }
  else return 0;
}

/* initialize matrix */
int zero_matrix(matrix_ptr m, long int row_len)
{
  long int i,j;

  if (row_len > 0) {
    m->len = row_len;
    for (i = 0; i < row_len*row_len; i++) {
      m->data[i] = (data_t)(IDENT);
    }
    return 1;
  }
  else return 0;
}

data_t *get_matrix_start(matrix_ptr m)
{
  return m->data;
}

/*************************************************/

/* mmm */
void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int length = get_matrix_row_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;

  for (i = 0; i < length; i++) {
    for (j = 0; j < length; j++) {
      sum = IDENT;
      for (k = 0; k < length; k++) {
        sum += a0[i*length+k] * b0[k*length+j];
	//printf("i: %d, j: %d, k: %d\n", i, j, k);
      }
      c0[i*length+j] += sum;
    }
  }
}

/* mmm */
void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int length = get_matrix_row_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t r;

  for (k = 0; k < length; k++) {
    for (i = 0; i < length; i++) {
      r = a0[i*length+k];
      for (j = 0; j < length; j++) {
        c0[i*length+j] += r*b0[k*length+j];
      }
    }
  }
}

/* mmm */
void mmm_jki(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int length = get_matrix_row_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t r;

  for (j = 0; j < length; j++) {
    for (k = 0; k < length; k++) {
      r = b0[k*length+j];
      for (i = 0; i < length; i++) {
        c0[i*length+j] += a0[i*length+k]*r;
      }
    }
  }
}

/* mmm blocking */
void bijk(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k, kk, jj;
  long int length = get_matrix_row_length(a);
  if(length%BLOCKS != 0)
  {
    printf("Matrix is not divisible by %d. Not running this computation.", BLOCKS);
    return;
  }
  int en = length;		/* assume that length is divisible by bsize*/
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;
  
  for(kk = 0; kk < en; kk += BLOCKS)
  {
    for(jj = 0; jj < en; jj += BLOCKS)
    {
      for(i = 0; i < length; i++)
      {
        for(j = jj; j < jj + BLOCKS; j++)
        {
	  sum = IDENT;
	  for(k = kk; k < kk + BLOCKS; k++)
	  {
	    sum += a0[i*length+k] * b0[k*length+j];
	    // printf("i: %d, j: %d, k: %d\n", i, j, k);
	  }
	  c0[i*length+j] = sum;
	}
      }
    }
  }
}

/* mmm blocking */
void bjki(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k, kk, ii;
  long int length = get_matrix_row_length(a);
  if(length%BLOCKS != 0)
  {
    printf("Matrix is not divisible by %d. Not running this computation.", BLOCKS);
    return;
  }
  int en = (length/BLOCKS);              /* assume that length is divisible by bsize*/
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;

  for(ii = 0; ii < en; ii += BLOCKS)
  {
    for(kk = 0; kk < en; kk += BLOCKS)
    {
      for(j = 0; j < length; j++)
      {
        for(k = kk; k < kk + BLOCKS; k++)
        {
          sum = IDENT;
          for(i = ii; i < ii + BLOCKS; i++)
          {
            sum += a0[j*length+i] * b0[i*length+k];
          }
          c0[j*length+k] = sum;
        }
      }
    }
  }
}

/* mmm blocking */
void bkij(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k, ii, jj;
  long int length = get_matrix_row_length(a);
  if(length%BLOCKS != 0)
  {
    printf("Matrix is not divisible by %d. Not running this computation.", BLOCKS);
    return;
  }
  int en = (length/BLOCKS);              /* assume that length is divisible by bsize*/
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;

  for(jj = 0; jj < en; jj += BLOCKS)
  {
    for(ii = 0; ii < en; ii += BLOCKS)
    {
      for(k = 0; k < length; k++)
      {
        for(i = ii; i < ii + BLOCKS; i++)
        {
          sum = IDENT;
          for(j = jj; j < jj + BLOCKS; j++)
          {
            sum += a0[k*length+j] * b0[j*length+i];
          }
          c0[k*length+i] = sum;
        }
      }
    }
  }
}

/* Lab 6 Q2 additions */
/* mmm blocking kij omp*/
void bkij_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k, ii, jj;
  long int length = get_matrix_row_length(a);
  if(length%BLOCKS != 0)
  {
    printf("Matrix is not divisible by %d. Not running this computation.", BLOCKS);
    return;
  }
  int en = (length/BLOCKS);              /* assume that length is divisible by bsize*/
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;
  
#pragma omp parallel for private(jj, ii, k, i, j) shared(a0, b0, c0, length)
  for(jj = 0; jj < en; jj += BLOCKS)
  {
    for(ii = 0; ii < en; ii += BLOCKS)
    {
      for(k = 0; k < length; k++)
      {
        for(i = ii; i < ii + BLOCKS; i++)
        {
          sum = IDENT;
          for(j = jj; j < jj + BLOCKS; j++)
          {
            sum += a0[k*length+j] * b0[j*length+i];
          }
          c0[k*length+i] = sum;
        }
      }
    }
  }
}
