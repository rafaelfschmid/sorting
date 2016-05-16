/**
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

////////////////////////////////////////////////////////////////////////////////
//
//  QUICKSORT.CU
//
//  Implementation of a parallel quicksort in CUDA. It comes in
//  several parts:
//
//  1. A small-set insertion sort. We do this on any set with <=32 elements
//  2. A partitioning kernel, which - given a pivot - separates an input
//     array into elements <=pivot, and >pivot. Two quicksorts will then
//     be launched to resolve each of these.
//  3. A quicksort co-ordinator, which figures out what kernels to launch
//     and when.
//
////////////////////////////////////////////////////////////////////////////////
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include "cdpQuicksort.h"

////////////////////////////////////////////////////////////////////////////////
// Inline PTX call to return index of highest non-zero bit in a word
////////////////////////////////////////////////////////////////////////////////
static __device__ __forceinline__ unsigned int __qsflo(unsigned int word)
{
    unsigned int ret;
    asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
//
//  ringbufAlloc
//
//  Allocates from a ringbuffer. Allows for not failing when we run out
//  of stack for tracking the offset counts for each sort subsection.
//
//  We use the atomicMax trick to allow out-of-order retirement. If we
//  hit the size limit on the ringbuffer, then we spin-wait for people
//  to complete.
//
////////////////////////////////////////////////////////////////////////////////
template< typename T >
static __device__ T *ringbufAlloc(qsortRingbuf *ringbuf)
{
    // Wait for there to be space in the ring buffer. We'll retry only a fixed
    // number of times and then fail, to avoid an out-of-memory deadlock.
    unsigned int loop = 10000;

    while (((ringbuf->head - ringbuf->tail) >= ringbuf->stacksize) && (loop-- > 0));

    if (loop == 0)
        return NULL;

    // Note that the element includes a little index book-keeping, for freeing later.
    unsigned int index = atomicAdd((unsigned int *) &ringbuf->head, 1);
    T *ret = (T *)(ringbuf->stackbase) + (index & (ringbuf->stacksize-1));
    ret->index = index;

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
//
//  ringBufFree
//
//  Releases an element from the ring buffer. If every element is released
//  up to and including this one, we can advance the tail to indicate that
//  space is now available.
//
////////////////////////////////////////////////////////////////////////////////
template< typename T >
static __device__ void ringbufFree(qsortRingbuf *ringbuf, T *data)
{
    unsigned int index = data->index;       // Non-wrapped index to free
    unsigned int count = atomicAdd((unsigned int *)&(ringbuf->count), 1) + 1;
    unsigned int max = atomicMax((unsigned int *)&(ringbuf->max), index + 1);

    // Update the tail if need be. Note we update "max" to be the new value in ringbuf->max
    if (max < (index+1)) max = index+1;

    if (max == count)
        atomicMax((unsigned int *)&(ringbuf->tail), count);
}

////////////////////////////////////////////////////////////////////////////////
//
//  qsort_warp
//
//  Simplest possible implementation, does a per-warp quicksort with no inter-warp
//  communication. This has a high atomic issue rate, but the rest should actually
//  be fairly quick because of low work per thread.
//
//  A warp finds its section of the data, then writes all data <pivot to one
//  buffer and all data >pivot to the other. Atomics are used to get a unique
//  section of the buffer.
//
//  Obvious optimisation: do multiple chunks per warp, to increase in-flight loads
//  and cover the instruction overhead.
//
////////////////////////////////////////////////////////////////////////////////
__global__ void qsort_warp(unsigned *indata,
                           unsigned *outdata,
                           unsigned int offset,
                           unsigned int len,
                           qsortAtomicData *atomicData,
                           qsortRingbuf *atomicDataStack,
                           unsigned int source_is_indata,
                           unsigned int depth)
{
    // Find my data offset, based on warp ID
    unsigned int thread_id = threadIdx.x + (blockIdx.x << QSORT_BLOCKSIZE_SHIFT);
    //unsigned int warp_id = threadIdx.x >> 5;   // Used for debug only
    unsigned int lane_id = threadIdx.x & (warpSize-1);

    // Exit if I'm outside the range of sort to be done
    if (thread_id >= len)
        return;

    //
    // First part of the algorithm. Each warp counts the number of elements that are
    // greater/less than the pivot.
    //
    // When a warp knows its count, it updates an atomic counter.
    //

    // Read in the data and the pivot. Arbitrary pivot selection for now.
    unsigned pivot = indata[offset + len/2];
    unsigned data  = indata[offset + thread_id];

    // Count how many are <= and how many are > pivot.
    // If all are <= pivot then we adjust the comparison
    // because otherwise the sort will move nothing and
    // we'll iterate forever.
    unsigned int greater = (data > pivot);
    unsigned int gt_mask = __ballot(greater);

    if (gt_mask == 0)
    {
        greater = (data >= pivot);
        gt_mask = __ballot(greater);    // Must re-ballot for adjusted comparator
    }

    unsigned int lt_mask = __ballot(!greater);
    unsigned int gt_count = __popc(gt_mask);
    unsigned int lt_count = __popc(lt_mask);

    // Atomically adjust the lt_ and gt_offsets by this amount. Only one thread need do this. Share the result using shfl
    unsigned int lt_offset, gt_offset;

    if (lane_id == 0)
    {
        if (lt_count > 0)
            lt_offset = atomicAdd((unsigned int *) &atomicData->lt_offset, lt_count);

        if (gt_count > 0)
            gt_offset = len - (atomicAdd((unsigned int *) &atomicData->gt_offset, gt_count) + gt_count);
    }

    lt_offset = __shfl((int)lt_offset, 0);   // Everyone pulls the offsets from lane 0
    gt_offset = __shfl((int)gt_offset, 0);

    __syncthreads();

    // Now compute my own personal offset within this. I need to know how many
    // threads with a lane ID less than mine are going to write to the same buffer
    // as me. We can use popc to implement a single-operation warp scan in this case.
    unsigned lane_mask_lt;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
    unsigned int my_mask = greater ? gt_mask : lt_mask;
    unsigned int my_offset = __popc(my_mask & lane_mask_lt);

    // Move data.
    my_offset += greater ? gt_offset : lt_offset;
    outdata[offset + my_offset] = data;


    // Count up if we're the last warp in. If so, then Kepler will launch the next
    // set of sorts directly from here.
    if (lane_id == 0)
    {
        // Count "elements written". If I wrote the last one, then trigger the next qsorts
        unsigned int mycount = lt_count + gt_count;

        if (atomicAdd((unsigned int *) &atomicData->sorted_count, mycount) + mycount == len)
        {
            // We're the last warp to do any sorting. Therefore it's up to us to launch the next stage.
            unsigned int lt_len = atomicData->lt_offset;
            unsigned int gt_len = atomicData->gt_offset;

            cudaStream_t lstream, rstream;
            cudaStreamCreateWithFlags(&lstream, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&rstream, cudaStreamNonBlocking);

            // Begin by freeing our atomicData storage. It's better for the ringbuffer algorithm
            // if we free when we're done, rather than re-using (makes for less fragmentation).
            ringbufFree<qsortAtomicData>(atomicDataStack, atomicData);

            // Exceptional case: if "lt_len" is zero, then all values in the batch
            // are equal. We are then done (may need to copy into correct buffer, though)
            if (lt_len == 0)
            {
                if (source_is_indata)
                    cudaMemcpyAsync(indata+offset, outdata+offset, gt_len*sizeof(unsigned), cudaMemcpyDeviceToDevice, lstream);

                return;
            }

            // Start with lower half first
            if (lt_len > BITONICSORT_LEN)
            {
                // If we've exceeded maximum depth, fall through to backup big_bitonicsort
                if (depth >= QSORT_MAXDEPTH)
                {
                    // The final bitonic stage sorts in-place in "outdata". We therefore
                    // re-use "indata" as the out-of-range tracking buffer. For (2^n)+1
                    // elements we need (2^(n+1)) bytes of oor buffer. The backup qsort
                    // buffer is at least this large when sizeof(QTYPE) >= 2.
                    big_bitonicsort<<< 1, BITONICSORT_LEN, 0, lstream >>>(outdata, source_is_indata ? indata : outdata, indata, offset, lt_len);
                }
                else
                {
                    // Launch another quicksort. We need to allocate more storage for the atomic data.
                    if ((atomicData = ringbufAlloc<qsortAtomicData>(atomicDataStack)) == NULL)
                        printf("Stack-allocation error. Failing left child launch.\n");
                    else
                    {
                        atomicData->lt_offset = atomicData->gt_offset = atomicData->sorted_count = 0;
                        unsigned int numblocks = (unsigned int)(lt_len+(QSORT_BLOCKSIZE-1)) >> QSORT_BLOCKSIZE_SHIFT;
                        qsort_warp<<< numblocks, QSORT_BLOCKSIZE, 0, lstream >>>(outdata, indata, offset, lt_len, atomicData, atomicDataStack, !source_is_indata, depth+1);
                    }
                }
            }
            else if (lt_len > 1)
            {
                // Final stage uses a bitonic sort instead. It's important to
                // make sure the final stage ends up in the correct (original) buffer.
                // We launch the smallest power-of-2 number of threads that we can.
                unsigned int bitonic_len = 1 << (__qsflo(lt_len-1U)+1);
                bitonicsort<<< 1, bitonic_len, 0, lstream >>>(outdata, source_is_indata ? indata : outdata, offset, lt_len);
            }
            // Finally, if we sorted just one single element, we must still make
            // sure that it winds up in the correct place.
            else if (source_is_indata && (lt_len == 1))
                indata[offset] = outdata[offset];

            if (cudaPeekAtLastError() != cudaSuccess)
                printf("Left-side launch fail: %s\n", cudaGetErrorString(cudaGetLastError()));


            // Now the upper half.
            if (gt_len > BITONICSORT_LEN)
            {
                // If we've exceeded maximum depth, fall through to backup big_bitonicsort
                if (depth >= QSORT_MAXDEPTH)
                    big_bitonicsort<<< 1, BITONICSORT_LEN, 0, rstream >>>(outdata, source_is_indata ? indata : outdata, indata, offset+lt_len, gt_len);
                else
                {
                    // Allocate new atomic storage for this launch
                    if ((atomicData = ringbufAlloc<qsortAtomicData>(atomicDataStack)) == NULL)
                        printf("Stack allocation error! Failing right-side launch.\n");
                    else
                    {
                        atomicData->lt_offset = atomicData->gt_offset = atomicData->sorted_count = 0;
                        unsigned int numblocks = (unsigned int)(gt_len+(QSORT_BLOCKSIZE-1)) >> QSORT_BLOCKSIZE_SHIFT;
                        qsort_warp<<< numblocks, QSORT_BLOCKSIZE, 0, rstream >>>(outdata, indata, offset+lt_len, gt_len, atomicData, atomicDataStack, !source_is_indata, depth+1);
                    }
                }
            }
            else if (gt_len > 1)
            {
                unsigned int bitonic_len = 1 << (__qsflo(gt_len-1U)+1);
                bitonicsort<<< 1, bitonic_len, 0, rstream >>>(outdata, source_is_indata ? indata : outdata, offset+lt_len, gt_len);
            }
            else if (source_is_indata && (gt_len == 1))
                indata[offset+lt_len] = outdata[offset+lt_len];

            if (cudaPeekAtLastError() != cudaSuccess)
                printf("Right-side launch fail: %s\n", cudaGetErrorString(cudaGetLastError()));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
//  run_quicksort
//
//  Host-side code to run the Kepler version of quicksort. It's pretty
//  simple, because all launch control is handled on the device via CDP.
//
//  All parallel quicksorts require an equal-sized scratch buffer. This
//  must be passed in ahead of time.
//
//  Returns the time elapsed for the sort.
//
////////////////////////////////////////////////////////////////////////////////
float run_quicksort_cdp(unsigned *gpudata, unsigned *scratchdata, unsigned int count, cudaStream_t stream)
{
    unsigned int stacksize = QSORT_STACK_ELEMS;

    // This is the stack, for atomic tracking of each sort's status
    qsortAtomicData *gpustack;
    checkCudaErrors(cudaMalloc((void **)&gpustack, stacksize * sizeof(qsortAtomicData)));
    checkCudaErrors(cudaMemset(gpustack, 0, sizeof(qsortAtomicData)));     // Only need set first entry to 0

    // Create the memory ringbuffer used for handling the stack.
    // Initialise everything to where it needs to be.
    qsortRingbuf buf;
    qsortRingbuf *ringbuf;
    checkCudaErrors(cudaMalloc((void **)&ringbuf, sizeof(qsortRingbuf)));
    buf.head = 1;           // We start with one allocation
    buf.tail = 0;
    buf.count = 0;
    buf.max = 0;
    buf.stacksize = stacksize;
    buf.stackbase = gpustack;
    checkCudaErrors(cudaMemcpy(ringbuf, &buf, sizeof(buf), cudaMemcpyHostToDevice));


    // Timing events...
    cudaEvent_t ev1, ev2;
    checkCudaErrors(cudaEventCreate(&ev1));
    checkCudaErrors(cudaEventCreate(&ev2));
    checkCudaErrors(cudaEventRecord(ev1));

    // Now we trivially launch the qsort kernel
    if (count > BITONICSORT_LEN)
    {
        unsigned int numblocks = (unsigned int)(count+(QSORT_BLOCKSIZE-1)) >> QSORT_BLOCKSIZE_SHIFT;
        qsort_warp<<< numblocks, QSORT_BLOCKSIZE, 0, stream >>>(gpudata, scratchdata, 0U, count, gpustack, ringbuf, true, 0);
    }
    else
    {
        bitonicsort<<< 1, BITONICSORT_LEN >>>(gpudata, gpudata, 0, count);
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(ev2));
    checkCudaErrors(cudaDeviceSynchronize());

    float elapse=0.0f;

    if (cudaPeekAtLastError() != cudaSuccess)
        printf("Launch failure: %s\n", cudaGetErrorString(cudaGetLastError()));
    else
        checkCudaErrors(cudaEventElapsedTime(&elapse, ev1, ev2));

    // Sanity check that the stack allocator is doing the right thing
    checkCudaErrors(cudaMemcpy(&buf, ringbuf, sizeof(*ringbuf), cudaMemcpyDeviceToHost));

    if (count > BITONICSORT_LEN && buf.head != buf.tail)
    {
        printf("Stack allocation error!\nRingbuf:\n");
        printf("\t head = %u\n", buf.head);
        printf("\t tail = %u\n", buf.tail);
        printf("\tcount = %u\n", buf.count);
        printf("\t  max = %u\n", buf.max);
    }

    // Release our stack data once we're done
    checkCudaErrors(cudaFree(ringbuf));
    checkCudaErrors(cudaFree(gpustack));

    return elapse;
}

static void usage()
{
    printf("Syntax: qsort [-size=<num>] [-seed=<num>] [-debug] [-loop-step=<num>] [-verbose]\n");
    printf("If loop_step is non-zero, will run from 1->array_len in steps of loop_step\n");
}

void cudaTest(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("cuda returned error %s (code %d), line(%d)\n",
				cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
}

void print(uint* host_data, uint n) {
	std::cout << "\n";
	for (uint i = 0; i < n; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << "\n";
}

#define ELAPSED_TIME 0

// Host side entry
int main(int argc, char *argv[])
{
	uint num_of_elements;
		uint i;

		scanf("%d", &num_of_elements);
		uint mem_size_vec = sizeof(int) * num_of_elements;
		uint *h_vec = (uint *) malloc(mem_size_vec);
		for (i = 0; i < num_of_elements; i++) {
			scanf("%d", &h_vec[i]);
		}

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		uint *d_scratchdata, *d_vec;

		cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
		cudaTest(cudaMalloc((void **) &d_scratchdata, mem_size_vec));
		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));

		cudaEventRecord(start);
        float elapse;
        elapse = run_quicksort_cdp(d_vec, d_scratchdata, num_of_elements, NULL);
		cudaEventRecord(stop);

		cudaError_t errSync = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess)
			printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

		cudaMemcpy(h_vec, d_vec, mem_size_vec, cudaMemcpyDeviceToHost);

		if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\n";
		} else
			print(h_vec, num_of_elements);

		free(h_vec);
		cudaFree(d_vec);
		cudaFree(d_scratchdata);
		cudaDeviceReset();

		return 0;
}




