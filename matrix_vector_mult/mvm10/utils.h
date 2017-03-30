/*
 * utils.h - A couple of helper functions
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <tmmintrin.h>

// Float computation

typedef union { __m128 v; float a[4]; } uf;
typedef union { __m256 v; float a[8]; } uf8;

void printVec4(__m128 v, char const * name)
{
  uf u;
  u.v = v;
  printf("Vector %s: [ %f\t%f\t%f\t%f ]\n", name, u.a[0], u.a[1], u.a[2], u.a[3]);
}

void printVec8(__m256 v, char const * name)
{
	uf8 u;
	u.v = v;
	printf("Vector %s: [ %f\t%f\t%f\t%f\t%f\t%f\t%f\t%f ]\n", name, u.a[0], u.a[1], u.a[2], u.a[3], u.a[4], u.a[5], u.a[6], u.a[7]);
}


void _printM(float const * const m, size_t const row, size_t const col, char const * name)
{
  int i;
  int j;
  printf("=========\n");
  printf("%s:\n", name);
  
  for (i = 0; i < row; ++i) {
      printf("[ ");	  
      for (j = 0; j < col; ++j) {
          printf("%f\t", m[i*col + j]);
      }
      printf(" ]\n");
  }
  printf("=========\n");
}

void setzero(float * m, size_t M, size_t N)
{
  int i;
  for (i = 0; i < M*N; ++i)  m[i] = 0;
}

void setrandom(float * m, size_t M, size_t N)
{
  int i;
  srand(time(NULL));  
  for (i = 0; i < M*N; ++i)  m[i] = (float)(rand())/RAND_MAX;;
}

// Double computation

typedef union { __m128 v; double a[2]; } ud;

void printVec2(__m128 v, char const * name)
{
  ud u;
  u.v = v;
  printf("Vector %s: [ %f\t%f ]\n", name, u.a[0], u.a[1]);
}


void _printMd(double const * const m, size_t const row, size_t const col, char const * name)
{
  int i,j;
  printf("=========\n");
  printf("%s:\n", name);  
  for (i = 0; i < row; ++i) {
      printf("[ ");
      for (j = 0; j < col; ++j) {
          printf("%f\t", m[i*col + j]);
      }
      printf(" ]\n");
  }
  printf("=========\n");
}

void setzerod(double * m, size_t M, size_t N)
{
  int i;
  for (i = 0; i < M*N; ++i)  m[i] = 0;
}

void setrandomd(double * m, size_t M, size_t N)
{
  int i;
  srand(time(NULL));
  for (i = 0; i < M*N; ++i)  m[i] = (double)(rand())/RAND_MAX;;
}


