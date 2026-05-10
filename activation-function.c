#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "activation-function.h"
#include "macro.h"

float leakyRelu(float x) {
  return x > 0 ? x : (0.01f * x);
}

float Relu(float x) {
  return MAX(x,0);
} 

float Relu_derivative(float x) {
	return x > 0 ? 1.0f : 0.0f;
}	

void softmax(float* prob_dist, float* output, int count) {
	float max = prob_dist[0];
	for (int i = 0; i < count; i++) {
		max = (prob_dist[i] > max) ? prob_dist[i] : max;
	}
	float total = 0.0f;
	for (int i = 0; i < count; i++) {
		output[i] = expf(prob_dist[i] - max);
		total += output[i];
	}
        for (int i = 0; i < 10; i++) {
                printf("%f, ", output[i]);
        }
        printf("\n");

	for (int i = 0; i < count; i++) {
		output[i] = output[i] / total;
	}
}

