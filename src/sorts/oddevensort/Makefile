TIME=1

all: sortingNetworks

bitonicSort.o: bitonicSort.cu
	nvcc -ccbin g++ -I/usr/local/cuda-7.5/samples/common/inc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o bitonicSort.o -c bitonicSort.cu

main.o: main.cpp
	nvcc -ccbin g++ -I/usr/local/cuda-7.5/samples/common/inc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o main.o -c main.cpp -DELAPSED_TIME=$(TIME)

oddEvenMergeSort.o: oddEvenMergeSort.cu
	nvcc -ccbin g++ -I/usr/local/cuda-7.5/samples/common/inc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o oddEvenMergeSort.o -c oddEvenMergeSort.cu

sortingNetworks: bitonicSort.o main.o oddEvenMergeSort.o
	nvcc -ccbin g++   -m64      -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o oddevensort.exe bitonicSort.o main.o oddEvenMergeSort.o

clean:
	rm -f oddevensort.exe bitonicSort.o main.o oddEvenMergeSort.o

