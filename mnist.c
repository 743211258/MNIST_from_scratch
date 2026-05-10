#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "activation-function.h"
#include "define-struct.h"
#include "loss-function.h"
#include "macro.h"

int arr[] = {784, 128, 64, 10};

int argmax(float* arr, int n) {
    int idx = 0;
    float max = arr[0];

    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
            idx = i;
        }
    }

    return idx;
}

void parse_csv(char* line, float* inputs, int* label) {
	char *token = strtok(line, ",");
	*label = atoi(token);
	for (int i = 0; i < 784; i++) {
		token = strtok(NULL, ",");
		inputs[i] = atof(token) / 255.0f;
	}
}

void network_forward(layer* layers, float* inputs, int number_of_layers) {
	for (int i = 0; i < number_of_layers; i++) {
		if (i == 0) {
			layers[i].forward(&layers[i], inputs, true);
		} else if (i < number_of_layers - 1){
			layers[i].forward(&layers[i], layers[i - 1].activated_output, true);
		} else {
			layers[i].forward(&layers[i], layers[i - 1].activated_output, false);
		}
	}
}

void backpropogate(layer* layers, float* inputs, float* outputs, float learning_rate, int number_of_layers, int label) {
	int out = layers[number_of_layers - 1].out;
	float* delta = malloc(out * sizeof(float));
	for (int i = 0; i < out; i++) {
		delta[i] = outputs[i];
	}
	delta[label] -= 1.0f;
	layers[number_of_layers - 1].backward(&layers[number_of_layers - 1], layers[number_of_layers - 2].activated_output, delta, learning_rate);
	free(delta);
	delta = layers[number_of_layers - 1].gradient;
	for (int i = number_of_layers - 2; i >= 0; i--) {
		float *current_inputs;
		if (i == 0) {
			current_inputs = inputs;
		} else {
			current_inputs = layers[i - 1].activated_output;
		}
		layers[i].backward(&layers[i], current_inputs, delta, learning_rate);
		delta = layers[i].gradient;
	}
}

float evaluate_accuracy(layer* layers, const char* path, int n_layers) {

    FILE* f = fopen(path, "r");
    if (f == NULL) {
        perror("Open test dataset failed!");
        exit(1);
    }

    char buffer[100000];
    float inputs[784];
    int label;

    int correct = 0;
    int total = 0;

    while (fgets(buffer, sizeof(buffer), f)) {

        parse_csv(buffer, inputs, &label);

        // forward
        network_forward(layers, inputs, n_layers);

        float outputs[10];
        softmax(layers[n_layers - 1].activated_output, outputs, 10);

        int pred = argmax(outputs, 10);

        if (pred == label) {
            correct++;
        }

        total++;
    }

    fclose(f);

    return (float)correct / total;
}


int main(int argc, char** argv) {
  layer* layers = malloc(3 * sizeof(layer));
	for (int i = 0; i < 3; i++) {
		layers[i] = (layer) {
			arr[i],
			arr[i+1],
			malloc(arr[i] * arr[i+1] * sizeof(float)),
			(float*) calloc(arr[i+1], sizeof(float)),
			malloc(arr[i+1] * sizeof(float)),
			malloc(arr[i+1] * sizeof(float)),
			malloc(arr[i] * sizeof(float)),
			forward,
			backward,
			he_init,
		};
		layers[i].he_init(&layers[i]);
	}
	FILE* train_fileptr = fopen("/home/ericzhang/Downloads/mnist_train.csv", "r");
	if (train_fileptr == NULL) {
		perror("Open the training dataset failed!");
		exit(1);
	}
	char buffer[100000];
	float inputs[784];
	int label;
	for (int i = 0; i < 10; i++) {
		rewind(train_fileptr);
		while (fgets(buffer, sizeof(buffer), train_fileptr)) {
			parse_csv(buffer, inputs, &label);
			network_forward(layers, inputs, 3);
			float* outputs = malloc(sizeof(float) * 10);
			softmax(layers[2].activated_output, outputs, 10);
			float loss = cross_entropy(outputs, label);
			printf("%f\n", loss);
			printf("-----------------------\n");
			backpropogate(layers, inputs, outputs, 1e-3f, 3, label);
			free(outputs);
		}
	}
	float acc = evaluate_accuracy(layers,
    "/home/ericzhang/Downloads/mnist_test.csv",
    3);

	printf("Test Accuracy: %f\n", acc);
  return 0;
}
