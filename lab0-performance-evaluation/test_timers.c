/************************************************************************
 To compile:
   gcc -O0 test_timers.c -lm -o test_timers
*/ 

#include <stdio.h>
#include <string.h>		// memset
#include <sys/time.h>		// get time of day
#include <sys/times.h>		// get time of day
#include <sys/mman.h>		// mmap
#include <unistd.h>		// getpid

// Used by gettimeofday --> NEED TO CHECK THIS AND POSSIBLY CHANGE FOR CORRECT TIMING
#define GET_TOD_TICS 1000000

// Used by RDTSC --> NEED TO CHECK THIS AND POSSIBLY CHANGE FOR CORRECT TIMING
#define CLK_RATE 3.0e9 

// Used by get_seconds --> NEED TO CHECK THIS AND POSSIBLY CHANGE FOR CORRECT TIMING
#define GET_SECONDS_TICS 100

struct timeval tim;    // used by gettimeofday

struct timezone tzp;

typedef union {        // used by RDTSC
  unsigned long long int64;
  struct {unsigned int lo, hi;} int32;
} tsc_counter;

/* We define RDTSC using inline assembly language instruction rdtsc */
#define RDTSC(cpu_c)              \
  __asm__ __volatile__ ("rdtsc" : \
  "=a" ((cpu_c).int32.lo),        \
  "=d"((cpu_c).int32.hi))

/* This "inline" version breaks on some compilers including recent MacOS gcc
 * inline double usecs_of_timeval(struct timeval * p_tv)
 * {
 *     return((double) p_tv->tv_sec) * 1e6 + ((double) p_tv->tv_usec);
 * } */
#define usecs_of_timeval(p_tv) \
           ( ((double) (p_tv)->tv_sec) * 1e6 + ((double) (p_tv)->tv_usec) )

double get_seconds() { 	/* routine to read time */
    struct tms rusage;
    times(&rusage);	/* UNIX utility: time in clock ticks */
    return (double)(rusage.tms_utime)/(double)(GET_SECONDS_TICS);
}

/*********************************************************************/
int main(int argc, char *argv[])
{
  long long int i, j, k, limit;
  long long int delta_int[20];
  long long int steps = 0;
  double sec0, sec,t1, t2, delta[10];	/* timing variables */
  struct timeval tv_start, tv_stop;     /* used by gettimeofday */
  tsc_counter t_0,t_1;                  /* used by RDTSC */
  struct timeval tv_wrap_start, tv_wrap_end;
  tsc_counter t_start, t_end;
  struct tms tic0, tic;
  double ticutime, timed;
#define BIG_POW_10 1000000000

  /* Test gettimeofday(). We try several powers of 10 as the number of
     iterations in the delay loop. Each time, call gettimeofday before and
     after the delay loop. Find the difference in units of microseconds,
     and save the result to print later. (The struct has two parts, seconds
     and microseconds, we use a call to "usecs_of_timeval()" to combine
     these together into a single number.) */
  gettimeofday(&tv_wrap_start, NULL);
  times(&tic0);

  printf("Using gettimeofday: \n");

  j = 0;
  for (limit = BIG_POW_10; limit >= 1; limit /= 10) {
    gettimeofday(&tv_start, NULL);
    for (i = 0; i < limit; i++) steps = steps + 1;
    gettimeofday(&tv_stop, NULL);
    delta[j++] = usecs_of_timeval(&tv_stop) - usecs_of_timeval(&tv_start);
  }

  for (i = 0; i < j; i++) 
    printf(" Time = %11.9f sec\n", delta[i]/(double)(GET_TOD_TICS));

  printf("gettimeofday tests done, %lld steps total\n", steps);
  printf("\n");


  /* Test RDTSC().  Try several powers of 10 as the number of iterations
     in the delay loop.  Each time, call RDTSC before and after the delay loop.
     Find the difference in the usecond part of the struct and save.
     NOTE: There have been problems with RDTSC, learn about them by searching
     online, but don't try to fix them here. */

  printf("Using RDTSC: \n");

  j = 0;
  for (limit = BIG_POW_10; limit >= 1; limit /= 10) {
    RDTSC(t_0);
    for (i = 0; i < limit; i++) steps = steps + 1;
    RDTSC(t_1);
    delta_int[j++] = t_1.int64-t_0.int64;
  }

  for (i = 0; i < j; i++)
    printf("Time = %11.9f sec (%lld cycles)\n",
	   (float)delta_int[i]/CLK_RATE ,delta_int[i]);

  printf("RDTSC tests done, %lld steps total\n", steps);
  printf("\n");


  /* Test times() through get_seconds(). As before, several runs with
     different powers of 10.     */

  printf("Using times(): \n");

  for (limit = BIG_POW_10; limit >= 1; limit /= 10) {
    sec0 = get_seconds();
    for (i = 0; i < limit; i++) steps = steps + 1;
    sec = (get_seconds() - sec0);	
    printf(" %10lld steps,  %11.8f sec\n", limit, sec);
  }

  printf("times() tests done, total steps = %lld \n", steps);
  printf("\n");

  times(&tic);
  gettimeofday(&tv_wrap_end, NULL);

  // delta_time: number of cycles for the program
  ticutime = (double) tic.tms_utime - tic0.tms_utime;

  // timed: time of program in secs
  timed = (usecs_of_timeval(&tv_wrap_end) - usecs_of_timeval(&tv_wrap_start))/(double)(GET_TOD_TICS);

  printf("The average number of ticks a second is: %12.8f\n", (float)ticutime/timed);
}
