export TIME=0
export BITS=10

all: generate radixSortCub radixSortThrust bitonic

generate: generate.cpp
	g++ -O3 generate.cpp -o generate.exe -DEXP_BITS_SIZE=$(BITS)
	g++ -O3 parser.cpp -o parser.exe -std=c++11

radixSortCub: radixSortCub.cu
	nvcc -O3 radixSortCub.cu -o radixSortCub.exe -I"../../lib/cub-1.5.2" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

radixSortThrust: radixSortThrust.cu
	nvcc -O3 radixSortThrust.cu -o radixSortThrust.exe -std=c++11 -DELAPSED_TIME=$(TIME)

bitonic: bitonic.cu
	nvcc -O3 bitonic.cu -o bitonic.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

