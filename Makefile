cc = gcc
CFLAGS = -g -Wall -Werror

TARGET = mnist

mnist: mnist.c define-struct.c activation-function.c loss-function.c
	$(cc) $(CFLAGS) $^ -o $@ -lm

clean:
	rm -rf mnist 

