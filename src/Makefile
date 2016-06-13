export TIME=1
export BITS=6

all: generate radixcub radixthrust mergemgpu

genrand: generate.cpp
	g++ -O3 generate.cpp -o generate.exe -DEXP_BITS_SIZE=$(BITS) -DRAND
	g++ -O3 parser.cpp -o parser.exe -std=c++11

genasc: generate.cpp
	g++ -O3 generate.cpp -o generate.exe -DEXP_BITS_SIZE=$(BITS) -DSORTASC

gendesc: generate.cpp
	g++ -O3 generate.cpp -o generate.exe -DEXP_BITS_SIZE=$(BITS) -DSORTDESC

radixcub: radixcub.cu
	nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 radixcub.cu -o radixcub.exe -I"../../segments/lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

radixthrust: radixthrust.cu
	nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 radixthrust.cu -o radixthrust.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

mergemgpu: mergemgpu.cu
	nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 mergemgpu.cu -o mergemgpu.exe -I"../../segments/lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

mergethrust: mergethrust.cu
	nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 mergethrust.cu -o mergethrust.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

radixthruststable: radixthrust_stable.cu
	nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 radixthrust_stable.cu -o radixthrust_stable.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

