/*
 * MVM 10x10
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <x86intrin.h>

#include "utils.h"
#include "rdtsc.h"


// Procedure mvm10: Serial code (do not need to modify)

void mvm10(float const * A, float const * x, float * y) {
  for(int i = 0; i < 10; i++) {
      float t = 0.f;
      for(int j = 0; j < 10; j++)
        t += A[i*10+j]*x[j];
      y[i] = t;
  }

}

// Procedure vec_mvm10: vector code
// Implement WITHOUT unaligned instructions
 
void vec_mvm10(float const * A, float const * x, float * y) {
  __m256i load_mask;
  load_mask=_mm256_set_epi32(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF);
  __m256i store_mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF);
  __m256 vec_1, vec_2;
  vec_1=_mm256_load_ps(x);
  vec_2=_mm256_maskload_ps(x+8, load_mask);

  __m256 data_1, data_2, tmp;
  for(int i=0; i<10; ++i){
    data_1=_mm256_loadu_ps(A+(i*10));
    data_2=_mm256_maskload_ps(A+(i*10)+8, load_mask);

    //Do the mul
    data_1=_mm256_mul_ps(data_1, vec_1);
    data_2=_mm256_mul_ps(data_2, vec_2);

    // Reduce
    tmp=_mm256_permute2f128_ps(data_1, data_1, 0x01);
    data_1=_mm256_hadd_ps(data_1, tmp);
    data_1=_mm256_hadd_ps(data_1, data_1);
    data_1=_mm256_shuffle_ps(data_1, data_2, 0x44);
    data_1=_mm256_hadd_ps(data_1, data_1);
    data_1=_mm256_hadd_ps(data_1, data_1);
    _mm256_maskstore_ps(y+i, store_mask, data_1);
  }

}

/*
 * Do not need to modify from here on
 */

#define RUNS     400
#define CYCLES_REQUIRED 1e7

void verify(float const * A, float const * x, float const * y)
{
  float * temp = (float *) _mm_malloc(sizeof(float)*10, 16);
  setzero(temp, 10, 1);
  printf("Verifying\n");
  for(int i = 0; i < 10; i++) {
      for(int j = 0; j < 10; j++)
        temp[i] += A[i*10+j]*x[j];
      double err = fabs(y[i] - temp[i]);
      if(err > 1E-5)
        {
          printf("Error at y[%d]\n", i);
        }
  }

  _mm_free(temp);

}

void test_vec_mvm10(float const * A, float const * x, float * y)
{
  tsc_counter start, end;
  double cycles = 0.;
  size_t num_runs = RUNS;

  //Cache warm-up
  // RDTSCP reads ts register guaranteeing that the execution of all the code
  // we wanted to measure is completed. This way we avoid including the
  // execution of a CPUID in between. The last CPUID guarantees no other
  // instruction can be scheduled before it (and so also before RDTSCP)

  CPUID(); RDTSC(start); CPUID(); RDTSC(end);
  CPUID(); RDTSC(start); CPUID(); RDTSC(end);
  CPUID(); RDTSC(start); CPUID(); RDTSC(end);

  while(1) {
      CPUID(); RDTSC(start);
      for (int i = 0; i < num_runs; ++i) {
          vec_mvm10(A, x, y);
      }
      CPUID(); RDTSC(end);

      cycles = (double)(COUNTER_DIFF(end, start));

      if(cycles >= CYCLES_REQUIRED) break;

      num_runs *= 2;

  }

  CPUID(); RDTSC(start);
  for (int i = 0; i < num_runs; ++i) {
      vec_mvm10(A, x, y);
  }
  CPUID(); RDTSC(end);

  cycles = (double)(COUNTER_DIFF(end, start))/num_runs;

  printf("Test vec_mvm10  - Performance [flops/cycle]: %f\n", 190/cycles);

#ifdef VERIFY
  verify(A, x, y);
#endif

}


int main()
{
  float * A = (float *) _mm_malloc(sizeof(float)*10*10, 32);
  float * x = (float *) _mm_malloc(sizeof(float)*10, 32);
  float * y = (float *) _mm_malloc(sizeof(float)*10, 32);

  setrandom(A, 10, 10);
  setrandom(x, 10, 1);

  test_vec_mvm10(A, x, y);

  return 0;
}
