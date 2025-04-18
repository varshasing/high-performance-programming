/****************************************************************************/
// gcc -O1 test_psum.c -lm -lrt -o test_psum

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#define MAX_SIZE 10000000

/* We want to test a variety of test sizes. We will generate these
   using the quadratic formula:  A x^2 + B x + C                     */
#define A   7  // coefficient of x^2
#define B  20  // coefficient of x
#define C 100  // constant term

#define NUM_TESTS 500   /* Number of different sizes to test */

/* Prototypes */
void psum1(float a[], float p[], long int n);
void psum2(float a[], float p[], long int n);


/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:
 
        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

/* method from test_clock_gettime, as told in instructions */
struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec) <0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}
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
    j *= 2;
  }
  return quasi_random;
}


/****************************************************************************/
int main(int argc, char *argv[])
{
  float *in, *out;
  long int x, n;
  double wd, intervals;
  struct timespec time_start, time_stop;
  struct timespec p1times[NUM_TESTS];
  struct timespec p2times[NUM_TESTS];
  long int xvals[NUM_TESTS];

  // initialize
  in = (float *) malloc(MAX_SIZE * sizeof(*in));
  out = (float *) malloc(MAX_SIZE * sizeof(*out));
  for (x = 0; x < MAX_SIZE; x++) {
    in[x] = (float)(x);
  }

  wd = wakeup_delay();

  /* process psum1 for various array sizes with psum1() and collect timing */
  /* ADD CODE to measure "start" time */
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<MAX_SIZE); x++) {
    /* ADD CODE to call psum1 and measure "stop" time */
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    psum1(in, out, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    p1times[x] = diff(time_start, time_stop);
    xvals[x] = n;
  }


  /* ADD CODE to repeat tests and measurements using psum2() */
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<MAX_SIZE); x++) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    psum2(in, out, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    p2times[x] = diff(time_start, time_stop);
  }

  /* output timing */
  printf("\n");
  printf("Time for running psum1 and psum2 per number of elements:\n");
  printf("n,psum1,psum2\n");
  /* ADD code to print out times for each value of n */
  for(x = 0; x<NUM_TESTS; x++) {;
    printf("%lld", xvals[x]);
    intervals = (((double)p1times[x].tv_sec) + ((double)p1times[x].tv_nsec)*1.0e-9);
    printf(",%lf", intervals);
    intervals = (((double)p2times[x].tv_sec) + ((double)p2times[x].tv_nsec)*1.0e-9);
    printf(",%lf\n", intervals);
  
  }

  /* Here we print things to prevent overzealous optimization */
  x = NUM_TESTS-1; n = A*x*x + B*x + C;
  if (n < MAX_SIZE) {
    printf("The biggest psum output value is: %d\n", out[n-1]);
  } else {
    printf("psum was not called for the %dth size because it is bigger than MAXSIZE\n", x+1);
  }
  printf("Wakeup delay calculated the value %f\n", wd);
} /* end of main() */


void psum1(float a[], float p[], long int n)
{
  long int i;

  p[0] = a[0];
  for (i = 1; i < n; i++) {
    p[i] = p[i-1] + a[i];
  }
}

void psum2(float a[], float p[], long int n)
{
  long int i;

  p[0] = a[0];
  for (i = 1; i < n-1; i+=2) {
    float mid_val = p[i-1] + a[i];
    p[i] = mid_val;
    p[i+1] = mid_val + a[i+1];
  }

  /* For odd n, finish remaining element */
  if (i < n) {
    p[i] = p[i-1] + a[i];
  }
}
