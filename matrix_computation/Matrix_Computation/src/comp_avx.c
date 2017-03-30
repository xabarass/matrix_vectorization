/**
 *      _________   _____________________  ____  ______
 *     / ____/   | / ___/_  __/ ____/ __ \/ __ \/ ____/
 *    / /_  / /| | \__ \ / / / /   / / / / / / / __/
 *   / __/ / ___ |___/ // / / /___/ /_/ / /_/ / /___
 *  /_/   /_/  |_/____//_/  \____/\____/_____/_____/
 *
 *  http://www.inf.ethz.ch/personal/markusp/teaching/
 *  How to Write Fast Numerical Code 263-2300 - ETH Zurich
 *  Copyright (C) 2017  Alen Stojanov   (astojanov@inf.ethz.ch)
 *                      Gagandeep Singh (gsingh@inf.ethz.ch)
 *                      Georg Ofenbeck  (ofenbeck@inf.ethz.ch)
 *	                Markus Pueschel (pueschel@inf.ethz.ch)
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see http://www.gnu.org/licenses/.
 */

#include "comp.h"

__attribute__((always_inline))
static inline avx_double sum_reduce_avx_value(avx_double value){
    avx_double tmp1, tmp2;
    tmp1 = _mm256_permute2f128_pd(value, value, 0x20);\
    tmp2 = _mm256_permute2f128_pd(value, value, 0x31);\
    tmp1=_mm256_hadd_pd(tmp1, tmp2);
    return _mm256_hadd_pd(tmp1, tmp1);
}

#define STORE_SUMS(__of, __sum_row){\
    sse_double res1=_mm256_extractf128_pd(sum##__sum_row##_1, 0);\
    sse_double res2=_mm256_extractf128_pd(sum##__sum_row##_2, 0);\
    sse_double res3=_mm256_extractf128_pd(sum##__sum_row##_3, 0);\
    sse_double res4=_mm256_extractf128_pd(sum##__sum_row##_4, 0);\
    _mm_store_sd(C+(n*(i+__of))+j, res1);\
    _mm_store_sd(C+(n*(i+__of))+j+1, res2);\
    _mm_store_sd(C+(n*(i+__of))+j+2, res3);\
    _mm_store_sd(C+(n*(i+__of))+j+3, res4);\
}

#define TRANSPOSE_AVX_DOUBLE(__row1, __row2, __row3, __row4){\
    tmp1 = _mm256_unpacklo_pd(__row1, __row2);\
    tmp2 = _mm256_unpackhi_pd(__row1, __row2);\
    tmp3 = _mm256_unpacklo_pd(__row3, __row4);\
    tmp4 = _mm256_unpackhi_pd(__row3, __row4);\
    \
    __row1 = _mm256_permute2f128_pd(tmp1, tmp3 , 0x20);\
    __row3 = _mm256_permute2f128_pd(tmp1, tmp3 , 0x31);\
    __row2 = _mm256_permute2f128_pd(tmp2, tmp4 , 0x20);\
    __row4 = _mm256_permute2f128_pd(tmp2, tmp4 , 0x31);\
    }

#define MUL_ROWS(__a_row, __b_row, \
                 __sum1, __sum2, __sum3, __sum4){\
    \
    /*Multiply rows with columns*/\
    /*sum[1,1]*/\
    tmp1=_mm256_mul_pd(__a_row, b_col1);\
    tmp2=_mm256_mul_pd(__b_row, a_col1);\
    /*sum[1,2]*/\
    tmp3=_mm256_mul_pd(__a_row, b_col2);\
    tmp4=_mm256_mul_pd(__b_row, a_col2);\
    /*sum[1,3]*/\
    tmp5=_mm256_mul_pd(__a_row, b_col3);\
    tmp6=_mm256_mul_pd(__b_row, a_col3);\
    /*sum[1,4]*/\
    tmp7=_mm256_mul_pd(__a_row, b_col4);\
    tmp8=_mm256_mul_pd(__b_row, a_col4);\
\
    /* filter minimum values*/\
    tmp1=_mm256_min_pd(tmp1, tmp2);\
    tmp3=_mm256_min_pd(tmp3, tmp4);\
    tmp5=_mm256_min_pd(tmp5, tmp6);\
    tmp7=_mm256_min_pd(tmp7, tmp8);\
\
    /* Sum up everything*/\
    tmp1=sum_reduce_avx_value(tmp1);\
    __sum1=_mm256_add_pd(__sum1, tmp1);\
\
    tmp3=sum_reduce_avx_value(tmp3);\
    __sum2=_mm256_add_pd(__sum2, tmp3);\
\
    tmp5=sum_reduce_avx_value(tmp5);\
    __sum3=_mm256_add_pd(__sum3, tmp5);\
\
    tmp7=sum_reduce_avx_value(tmp7);\
    __sum4=_mm256_add_pd(__sum4, tmp7);\
}\

//write the AVX version here, like the SSE, try to optimize as much as possible
void comp_avx(double *C , double *A , double *B , int n){
	int i, j, k;

    avx_double tmp1;
    avx_double tmp2;
    avx_double tmp3;
    avx_double tmp4;
    avx_double tmp5;
    avx_double tmp6;
    avx_double tmp7;
    avx_double tmp8;

    avx_double sum1_1;
    avx_double sum1_2;
    avx_double sum1_3;
    avx_double sum1_4;
    avx_double sum2_1;
    avx_double sum2_2;
    avx_double sum2_3;
    avx_double sum2_4;
    avx_double sum3_1;
    avx_double sum3_2;
    avx_double sum3_3;
    avx_double sum3_4;
    avx_double sum4_1;
    avx_double sum4_2;
    avx_double sum4_3;
    avx_double sum4_4;

    avx_double a_row1;
    avx_double a_row2;
    avx_double a_row3;
    avx_double a_row4;
    avx_double b_row1;
    avx_double b_row2;
    avx_double b_row3;
    avx_double b_row4;

    avx_double a_col1;
    avx_double a_col2;
    avx_double a_col3;
    avx_double a_col4;
    avx_double b_col1;
    avx_double b_col2;
    avx_double b_col3;
    avx_double b_col4;

    for (i = 0; i < n; i+=4){
        for (j = 0; j < n; j+=4){
            sum1_1=_mm256_setzero_pd();
            sum1_2=_mm256_setzero_pd();
            sum1_3=_mm256_setzero_pd();
            sum1_4=_mm256_setzero_pd();
            sum2_1=_mm256_setzero_pd();
            sum2_2=_mm256_setzero_pd();
            sum2_3=_mm256_setzero_pd();
            sum2_4=_mm256_setzero_pd();
            sum3_1=_mm256_setzero_pd();
            sum3_2=_mm256_setzero_pd();
            sum3_3=_mm256_setzero_pd();
            sum3_4=_mm256_setzero_pd();
            sum4_1=_mm256_setzero_pd();
            sum4_2=_mm256_setzero_pd();
            sum4_3=_mm256_setzero_pd();
            sum4_4=_mm256_setzero_pd();

            for (k = 0; k < n; k+=4){
                //For first multiplier
                //Load A row
                a_row1=_mm256_loadu_pd(A+(n*i)+k);
                a_row2=_mm256_loadu_pd(A+(n*(i+1))+k);
                a_row3=_mm256_loadu_pd(A+(n*(i+2))+k);
                a_row4=_mm256_loadu_pd(A+(n*(i+3))+k);
                //Load B row
                b_row1=_mm256_loadu_pd(B+(n*i)+k);
                b_row2=_mm256_loadu_pd(B+(n*(i+1))+k);
                b_row3=_mm256_loadu_pd(B+(n*(i+2))+k);
                b_row4=_mm256_loadu_pd(B+(n*(i+3))+k);

                //For second multiplier, load and transpose
                //Matrix A
                a_col1=_mm256_loadu_pd(A+(n*k)+j);
                a_col2=_mm256_loadu_pd(A+(n*(k+1))+j);
                a_col3=_mm256_loadu_pd(A+(n*(k+2))+j);
                a_col4=_mm256_loadu_pd(A+(n*(k+3))+j);
                TRANSPOSE_AVX_DOUBLE(a_col1, a_col2, a_col3, a_col4);

                //Matrix B
                b_col1=_mm256_loadu_pd(B+(n*k)+j);
                b_col2=_mm256_loadu_pd(B+(n*(k+1))+j);
                b_col3=_mm256_loadu_pd(B+(n*(k+2))+j);
                b_col4=_mm256_loadu_pd(B+(n*(k+3))+j);
                TRANSPOSE_AVX_DOUBLE(b_col1, b_col2, b_col3, b_col4);

                // Sums [1,1] [1,2] [1,3] [1,4]
                MUL_ROWS(a_row1, b_row1, sum1_1, sum1_2, sum1_3, sum1_4);
                // Sums [2,1] [2,2] [2,3] [2,4]
                MUL_ROWS(a_row2, b_row2, sum2_1, sum2_2, sum2_3, sum2_4);
                // Sums [3,1] [3,2] [3,3] [3,4]
                MUL_ROWS(a_row3, b_row3, sum3_1, sum3_2, sum3_3, sum3_4);
                // Sums [4,1] [4,2] [4,3] [4,4]
                MUL_ROWS(a_row4, b_row4, sum4_1, sum4_2, sum4_3, sum4_4);
            }

            STORE_SUMS(0, 1);
            STORE_SUMS(1, 2);
            STORE_SUMS(2, 3);
            STORE_SUMS(3, 4);
		}
	}
}
