#ifndef DEFINE_STRUCT_H
#define DEFINE_STRUCT_H

typedef struct layer layer;
typedef void(*forward_funptr)(layer* self, float* input, bool is_relu);
typedef void(*backward_funptr)(layer* self, float* inputs, float* delta, float learning_rate);
typedef void(*he_init_funptr)(layer* self);
void forward(layer* self, float* input, bool is_relu);
void backward(layer *self, float* inputs, float* delta, float learning_rate);
void he_init(layer* self);

struct layer {
	int in;
	int out;
	float* weights;
	float* bias;
	float* unactivated_output;
	float* activated_output;
	float* gradient;
	forward_funptr forward;
	backward_funptr backward;
	he_init_funptr he_init;
};

#endif
