#include <math.h>
#include <stdio.h>

#include "loss-function.h"

float cross_entropy(float* prob_dist, int label) {
	for (int i = 0; i < 10; i++) {
		printf("%f, ", prob_dist[i]);
	}
	printf("\n");
	float small_number = 1e-8f;
	return -logf(prob_dist[label] + small_number);
}
