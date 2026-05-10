#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "define-struct.h"
#include "activation-function.h"

void forward(layer* self, float* inputs, bool is_relu) {
	for (int i = 0; i < self->out; i++) {
		float sum = self->bias[i];
		for (int j = 0; j < self->in; j++) {
			sum += self->weights[i * self->in + j] * inputs[j];
		}
		self->unactivated_output[i] = sum;
		if (is_relu) {
			self->activated_output[i] = Relu(sum);
		} else {
			self->activated_output[i] = sum;	
		}
	}
}

void backward(layer* self, float* inputs, float* delta, float learning_rate) {
	for (int j = 0; j < self->in; j++) {
		float sum = 0.0f;
		for (int i = 0; i < self->out; i++) {
			sum += delta[i] * self->weights[i * self->in + j];
		}
		self->gradient[j] = sum * Relu_derivative(inputs[j]);
	}
	for (int i = 0; i < self->out; i++) {
		for (int j = 0; j < self->in; j++) {
			self->weights[i * self->in + j] -= learning_rate * delta[i] * inputs[j];
		}
		self->bias[i] -= learning_rate * delta[i];
	}
}

void box_muller(float* one, float* two, float sigma, float mu) {
	const float two_pi = 2 * (float) M_PI;
	float rand_one;
	float rand_two;
	do {
		rand_one = (float) rand() / (float) RAND_MAX;
	} while (rand_one == 0.0f);
	rand_two = (float) rand() / (float) RAND_MAX;

	float magnitude = sigma * sqrtf(-2.0f * logf(rand_one));
	*one = magnitude * cosf(two_pi * rand_two) + mu;
	*two = magnitude * sinf(two_pi * rand_two) + mu;									
}

void he_init(layer* self) {
	int total_nodes = self->in * self->out;
	float sigma = sqrtf(2.0f / (float) self->in);
	for (int i = 0; i < total_nodes / 2; i++) {
		box_muller(&self->weights[2 * i], &self->weights[2 * i + 1], sigma, 0.0f);
	}
	if (total_nodes % 2 == 1) {
		float temp;
		box_muller(&self->weights[total_nodes - 1], &temp, sigma, 0.0f);
	}
}
