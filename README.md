# MNIST_from_scratch
A simple fully connected neural network built from scratch (with some help from ChatGPT) using C language. The network was trained on MNIST dataset and achieved a 97.5% accuracy.

I built this project to help me understand the fundamentals of a neural network (forward, backpropogation, mostly the math behind it.

## Component
***activation-function.c:*** Implementations of a few activation functions.  
***define-struct.c:*** Implementations of a few functions for the struct layer.  
***loss-function.c:*** Implementation of the cross entropy loss function.  
***mnist.c:*** The main body of the neural network, including the initialization, training process, and evaluation.  

## Run the program
**Dependency:** gcc (Version 16.1.1 on my computer, but any older version will work as well).

To fetch the source code, run:  
```bash
git clone https://github.com/743211258/MNIST_from_scratch.git
```

To compile the source code, run:
```bash
cd ./MNIST_from_scratch
make
```

To run the source code, run:
```bash
./mnist
```

## A noticeable design choice

Since a neural network is a collection of nodes, it may seems natural to define a struct node and build a neural network out of it. However, I soon figured out that this approach has worse cache locality and optimization. Given the fact that most operations are performing the same calculation on all nodes, it is reasonable to store those data in cache and vectorize those operations like SIMD. If you store data in nodes and an operation only requires part of them, then putting a whole node in cache is a complete waste of space, and non-consecutive data across struct node will slow up SIMD due to slower vector load.

## License
MIT
