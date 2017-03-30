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

//scalar code for the matrix computation
void comp_sisd(double *C , double *A , double *B , int n){
	int i, j, k;
	double sum;
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			sum = 0;
			for (k = 0; k < n; k++){
				sum = sum + fmin(A[n*i+k]*B[n*k+j],B[n*i+k]*A[n*k+j]);
			}
			C[n*i+j] = sum;
		}
	}
}
