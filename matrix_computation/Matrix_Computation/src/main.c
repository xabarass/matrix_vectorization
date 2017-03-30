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

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include "comp.h"
#include "perf.h"

#define MAX_SIZE 1500

int cmp_matrices (double * A, double * B, int n) {
    int errorPos = -1;
    int i;
    for (i = 0; i < n * n; i++) {
        if (((uint64_t *) A)[i] != ((uint64_t *) B)[i]) {
            errorPos = i;
            break;
        }
    }
    return errorPos;
}

void validate(double *C, double *A, double * B, int n){
    int runs = 10;
    int j;
    int64_t sisd_cycles = 0, sse_cycles = 0, avx_cycles;
    double * C_valid = (double *) malloc(sizeof(double) * n * n);
    // Run the Scalar version
    cycles_count_start ();
    for (j = 0; j < runs; ++j) {
         comp_sisd(C, A, B, n);
    }
    sisd_cycles = cycles_count_stop()/runs;
    memcpy(C_valid, C, sizeof(double) * n * n);
    double flops = (double)n;
    flops = 4*flops*flops*flops;
    printf("N: %d SISD %g ",n, flops / sisd_cycles);

    // Run the SSE version
    cycles_count_start();
    for (j = 0; j < runs; ++j) {
         comp_sse(C, A, B, n);
    }
    sse_cycles = cycles_count_stop()/runs;
    int cmp = cmp_matrices(C, C_valid, n);
    if (cmp == -1) {
        printf("SSE %g ", flops / sse_cycles);
    } else {
        printf("SSE Fail ");
	printf("(index=> i: %d j: %d, SISD: %g SSE: %g) ",cmp/n,cmp%n,C_valid[cmp],C[cmp]);
    }

     // Run the AVX version
    cycles_count_start();
    for (j = 0; j < runs; ++j) {
         comp_avx(C, A, B, n);
    }
    avx_cycles = cycles_count_stop()/runs;
    cmp = cmp_matrices(C, C_valid, n);
    if (cmp == -1) {
        printf("AVX %g\n", flops / avx_cycles);
    } else {
        printf("AVX Fail ");
	printf("index=> i: %d j: %d, SISD: %g AVX: %g\n",cmp/n,cmp%n,C_valid[cmp],C[cmp]);
    }

    free(C_valid);
    
}

void rand_init(double * A, int n){
	int i;
	for(i=0; i < n*n; i++){
        A[i] = (double)(rand()%10+1);
	}
}

int main(){
	double *A, *B, *C;
	int n;
	A = (double *)malloc(MAX_SIZE*MAX_SIZE*sizeof(double));
	B = (double *)malloc(MAX_SIZE*MAX_SIZE*sizeof(double));
    C = (double *)malloc(MAX_SIZE*MAX_SIZE*sizeof(double));
	perf_init ();
    for(n=100; n <= 1500; n=n+100){
		rand_init(A,n);
		rand_init(B,n);
		validate(C,A,B,n);
	}
	perf_done();
	free(A);
	free(B);
	free(C);
}
