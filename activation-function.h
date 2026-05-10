#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

float leakyRelu(float x);

float Relu(float x);

float Relu_derivative(float x);

void softmax(float* prob_dist, float* output, int count);

#endif
