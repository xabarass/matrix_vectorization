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

static void print_matrix(double *m, int n, const char* name){
    printf(name);
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            printf("%g\t",m[i*n+j]);
        }
        printf("\n");
    }
}

static void print_vector(sse_double vector){
    static double buff[2];
    _mm_storeu_pd(buff, vector);
    printf("%g , %g\n", buff[0], buff[1]);
}

// write the SSE version here, try to optimize it as much as possible
void comp_sse(double *C , double *A , double *B , int n){

    memset(C, 0, n*n*sizeof(double));
	int i, j, k;
    sse_double sum1;
    sse_double sum2;
    sse_double sum3;
    sse_double sum4;

    sse_double tmp1;
    sse_double tmp2;
    sse_double tmp3;
    sse_double tmp4;

    sse_double a_row1;
    sse_double a_row2;
    sse_double b_row1;
    sse_double b_row2;

    sse_double a_col1;
    sse_double a_col2;
    sse_double b_col1;
    sse_double b_col2;

    for (i=0; i<n; i+=2){
        for (j=0; j<n; j+=2){
            sum1=_mm_setzero_pd();
            sum2=_mm_setzero_pd();
            sum3=_mm_setzero_pd();
            sum4=_mm_setzero_pd();

            for (k=0;k<n;k+=2){
                //Load first first multiplier
                a_row1=_mm_load_pd(A+(n*i)+k);
                a_row2=_mm_load_pd(A+(n*(i+1))+k);
                b_row1=_mm_load_pd(B+(n*i)+k);
                b_row2=_mm_load_pd(B+(n*(i+1))+k);

                //Load first second multiplier
                tmp1=_mm_load_pd(A+(n*k)+j);
                tmp2=_mm_load_pd(A+(n*(k+1))+j);
                tmp3=_mm_load_pd(B+(n*k)+j);
                tmp4=_mm_load_pd(B+(n*(k+1))+j);

                //Transpose second block
                a_col1=_mm_unpacklo_pd(tmp1, tmp2);
                a_col2=_mm_unpackhi_pd(tmp1, tmp2);
                b_col1=_mm_unpacklo_pd(tmp3, tmp4);
                b_col2=_mm_unpackhi_pd(tmp3, tmp4);

                //Multiplication first row with two columns
                tmp1=_mm_mul_pd(a_row1, b_col1);
                tmp2=_mm_mul_pd(a_row1, b_col2);
                tmp3=_mm_mul_pd(b_row1, a_col1);
                tmp4=_mm_mul_pd(b_row1, a_col2);
                //Take minimums
                tmp1=_mm_min_pd(tmp1, tmp3);
                tmp2=_mm_min_pd(tmp2, tmp4);
                //sum up everything
                tmp1=_mm_hadd_pd(tmp1, tmp1);
                tmp2=_mm_hadd_pd(tmp2, tmp2);
                sum1=_mm_add_pd(sum1, tmp1);
                sum2=_mm_add_pd(sum2, tmp2);

                //again, multiply
                tmp1=_mm_mul_pd(a_row2, b_col1);
                tmp2=_mm_mul_pd(a_row2, b_col2);
                tmp3=_mm_mul_pd(b_row2, a_col1);
                tmp4=_mm_mul_pd(b_row2, a_col2);
                //filter min
                tmp1=_mm_min_pd(tmp1, tmp3);
                tmp2=_mm_min_pd(tmp2, tmp4);
                //reduce
                tmp1=_mm_hadd_pd(tmp1, tmp1);
                tmp2=_mm_hadd_pd(tmp2, tmp2);
                sum3=_mm_add_pd(sum3, tmp1);
                sum4=_mm_add_pd(sum4, tmp2);
            }

            _mm_store_sd(C+(n*i)+j, sum1);
            _mm_store_sd(C+(n*i)+j+1, sum2);
            _mm_store_sd(C+(n*(i+1))+j, sum3);
            _mm_store_sd(C+(n*(i+1))+j+1, sum4);
        }
    }

}
