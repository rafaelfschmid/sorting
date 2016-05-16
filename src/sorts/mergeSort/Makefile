all: sortingNetworks

bitonic.o: bitonic.cu
	nvcc -ccbin g++ -I/usr/local/cuda-7.5/samples/common/inc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o bitonic.o -c bitonic.cu

main.o: main.cpp
	nvcc -ccbin g++ -I/usr/local/cuda-7.5/samples/common/inc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o main.o -c main.cpp

mergeSort.o: mergeSort.cu
	nvcc -ccbin g++ -I/usr/local/cuda-7.5/samples/common/inc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o mergeSort.o -c mergeSort.cu

sortingNetworks: bitonic.o main.o mergeSort.o
	nvcc -ccbin g++   -m64      -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o sortingNetworks bitonic.o main.o mergeSort.o

clean:
	rm -f sortingNetworks bitonic.o main.o mergeSort.o

