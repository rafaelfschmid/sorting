#include <time.h>
#include <algorithm>
#include <math.h>
#include <cstdlib>
#include <stdio.h>

#ifndef EXP_BITS_SIZE
#define EXP_BITS_SIZE 10
#endif

void vectors_gen(int num_elements, int bits_size_elements) {
	for (int i = 0; i < num_elements; i++)
	{
		printf("%d ", rand() % bits_size_elements);
	}
}

int main(int argc, char** argv) {

	if (argc < 2) {
		printf(
				"Parameters needed: <number of segments> \n\n");
		return 0;
	}

	int number_of_elements = atoi(argv[1]);

	srand(time(NULL));
	printf("%d\n", number_of_elements);
	vectors_gen(number_of_elements, pow(2, EXP_BITS_SIZE));
	printf("\n");

	return 0;
}

