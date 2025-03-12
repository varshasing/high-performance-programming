/*                         mem_bench.c
 *
 * Self-adjusting single-threaded memory bandwidth benchmark
 *
 * To compile:
 *
 *     gcc -O1 mem_bench.c -o mb
 */

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

double get_time()
{
  struct timespec ts;
  double rv;

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  rv = ((double)ts.tv_sec) + ((double)1.0e-9) * ((double)ts.tv_nsec);
  return(rv);
}

double * src;
double * dst;    // input and output arrays

double bench1(long inner_limit, long outer_limit)
{
  long i, j;
  double t_start, t_end;

  t_start = get_time();
  for(i=0; i<outer_limit; i++) {
    for(j=0; j<inner_limit; j++) {
      dst[j] = src[j];
    }
  }
  t_end = get_time();
  return(t_end - t_start);
}

int main(int argc, char ** argv)
{
  double bench_time;
  long array_size, loop_iters;
  long mem_alloc;

  mem_alloc = 1024L * 1024L;
  if (argc > 1) {
    sscanf(argv[1], "%ld", &mem_alloc);
  }
  /* must be a nontrivial size */
  if (mem_alloc < 256) {
    mem_alloc = 256;
  }
  /* must be a multiple of sizeof(double) */
  mem_alloc = sizeof(double) * (mem_alloc / sizeof(double));
  /* printf("mem_alloc = %ld\n", mem_alloc); */

  array_size = mem_alloc / sizeof(double);
  src = (double *) malloc((size_t) mem_alloc);
  if (src == 0) {
    fprintf(stderr, "Failed to allocate %ld bytes for src array\n", mem_alloc);
    exit(-1);
  }
  dst = (double *) malloc((size_t) mem_alloc);
  if (dst == 0) {
    fprintf(stderr, "Failed to allocate %ld bytes for dst array\n", mem_alloc);
    exit(-1);
  }

  bench_time = 0.0;
  loop_iters = 1;
  while (bench_time < 1.0) {
    loop_iters = loop_iters * 2;
    bench_time = bench1(array_size, loop_iters);
  }

  /* Now we know that loop_iters is a number of loop iterations that
     takes between 1 and 2 seconds to run. Next we need to delay a little
     to make sure the other copies of this benchmark all have a chance
     to do the same thing. */
  bench1(array_size, loop_iters);

  /* Now run the test again, this is the measurement we actually keep. */
  bench_time = bench1(array_size, loop_iters);

  /* We might be finishing sooner than the other tasks, but we want to
     make sure the computer is equally busy while they're doing their
     measurement. So we run the test one more time to keep CPU load high. */
  bench1(array_size, loop_iters);

  /* Compute bytes/sec and print the result. Size of memory allocation
     in bytes, times 2 (because we read that much, then write that much)
     divided by the time in seconds = bytes/sec */
  printf("memsize %10ld   %9.5g bytes/sec\n", mem_alloc,
             ((double)loop_iters) * ((double)mem_alloc) * 2.0 / bench_time);

  return 0;
}
