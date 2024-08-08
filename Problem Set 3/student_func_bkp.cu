/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <algorithm>
#include "stdio.h"

#include <iostream>
#include <cmath>

//const int BLOCK_SIZE = 512;
const int BLOCK_SIZE = 1024;

__device__
float min_k(float a, float b){
   return a < b ? a : b;
}

__device__
float max_k(float a, float b){
   return a > b ? a : b;
}

//KERNEL to reduce in a block and storing the result at the blockIdx location in global memory in d_out array
//Since, max_reduce would be almost similar, we incorporate a bool to make the switch in the associative operator, the writing back of result part.
__global__
void min_max_reduce(float* d_out , const float* const d_in, int input_size, bool isMin){

   unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int tid = threadIdx.x;

   extern __shared__ float sh_data[];

   // Transfer the data for this block to shared memory and pad for elements which are not present;
   if (input_id >= input_size){
      //PAD with data which doesn't change the result
      sh_data[tid] = d_in[0];
   } else {
      sh_data[tid] = d_in[input_id];
   }
   
   __syncthreads();// All data should be in shared memory block, before proceeding.

   for (int size = blockDim.x/2; size > 0; size>>=1){
      if (tid < size){
         sh_data[tid] = isMin ? min_k(sh_data[tid], sh_data[tid+size]) : max_k(sh_data[tid], sh_data[tid+size]);
      }
      __syncthreads();//Every step should be in sync to avoid reading and writing wrong values!
   }

   if (tid==0){
      d_out[blockIdx.x] = sh_data[tid];// thread ZERO would have the final result of reduce;
   }

}

unsigned int next_power_of_two(unsigned int n){
   unsigned count = 0;  
      
   // First n in the below condition  
   // is for the case where n is 0  
   if (n && !(n & (n - 1)))  
      return n;  
      
   while( n != 0)  
   {  
         n >>= 1;  
         count += 1;  
   }  
      
   return 1 << count;  
}

//Since, max_reduce would be almost similar, we incorporate a bool to make the switch in the associative operator.
float min_max_reduce(const float* const d_logLuminance, int input_size, bool isMin){
   int problem_size = input_size;
   float *d_intermediate = NULL;

   unsigned int n_blocks = ceil(1.0f*problem_size/BLOCK_SIZE);
   unsigned int threads = BLOCK_SIZE;
   unsigned int SH_MEM_BYTES = threads * sizeof(float);

   while (true) {
      float* d_blockMin = NULL; //For Storing Intermediate results for a block
      checkCudaErrors(cudaMalloc(&d_blockMin,   sizeof(float) * n_blocks));

      if (d_intermediate == NULL){
         //Launching kernels for the first time; Totally unavoidable step
         min_max_reduce<<<n_blocks, threads, SH_MEM_BYTES>>>(d_blockMin, d_logLuminance, problem_size, isMin);
      } else {
         min_max_reduce<<<n_blocks, threads, SH_MEM_BYTES>>>(d_blockMin, d_intermediate, problem_size, isMin);
      }
      //For Next Iteration
      problem_size = n_blocks;
      d_intermediate = d_blockMin;
      
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      if (problem_size == 1){
         float h_min;
         //Only a single block was launched. d_blockMin[0] should have answer.
         //cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
         checkCudaErrors(cudaMemcpy(&h_min, d_intermediate, sizeof(float), cudaMemcpyDeviceToHost));
         checkCudaErrors(cudaFree(d_intermediate));
         return h_min;
      } else if (problem_size > BLOCK_SIZE){
         // STILL MORE THAN ONE BLOCK REQUIRED
         n_blocks = ceil(1.0f*problem_size/BLOCK_SIZE); // BLOCKS to launch for next iteration
      } else if (problem_size <= BLOCK_SIZE){
         // Only ONE BLOCK is sufficient; But, number of threads can be lowered. By a simple computation.
         int nextPowerOfTwo = next_power_of_two(problem_size);
         threads = nextPowerOfTwo;
         n_blocks = 1;
         SH_MEM_BYTES = threads * sizeof(float);
      }
   }

}

//Histogram which uses atomic adds. Can we do better?
__global__
void histogram_atomic(unsigned int* d_histo_out, const float* const d_input_array, int input_size, float lumMin, float lumRange, unsigned int numBins){
   unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;//mapping to input data

   if (input_id > input_size) {
       return;
   }

   //bin = (lum[i] - lumMin) / lumRange * numBins; calculating the bin to which
   unsigned int bin = ((d_input_array[input_id] - lumMin)/lumRange)*numBins;
   if (bin == numBins) {// only in case for pixels with maximum intensity
       bin = numBins -1;
   }
   atomicAdd(&d_histo_out[bin],1);
}

__global__
void histogram_atomic_shared(unsigned int* d_histo_out, const float* const d_input_array, int input_size, float lumMin, float lumRange, unsigned int numBins){
   unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;//mapping to input data

   extern __shared__ unsigned int sh_data_array[];

   if (input_id > input_size) {
       return;
   }

//   if (input_id < numBins){
   if (threadIdx.x < numBins){
//      sh_data_array[input_id] = d_histo_out[input_id];
//      sh_data_array[threadIdx.x] = d_histo_out[input_id];
      sh_data_array[threadIdx.x] = 0;
   }
   __syncthreads();

   //bin = (lum[i] - lumMin) / lumRange * numBins; calculating the bin to which
   unsigned int bin = ((d_input_array[input_id] - lumMin)/lumRange)*numBins;
   if (bin == numBins) {// only in case for pixels with maximum intensity
       bin = numBins -1;
   }
   
   atomicAdd(&sh_data_array[bin],1);
   
   __syncthreads();
   
//   if (input_id < numBins){
   if (threadIdx.x < numBins){
//      atomicAdd(&d_histo_out[input_id], sh_data_array[input_id]);
      atomicAdd(&d_histo_out[threadIdx.x], sh_data_array[threadIdx.x]);
      //d_histo_out[input_id] = sh_data_array[input_id];
   }
   //atomicAdd(&d_histo_out[bin],1);
   __syncthreads();
}

//Histogram with simulated atomic additions. -- https://pdfs.semanticscholar.org/2325/0770d034de0602586dc039fe1c24a6b070a8.pdf
__global__
void histogram_simulated_atomic(unsigned int* d_histo_out, const float* const d_input_array, int input_size, float lumMin, float lumRange, unsigned int numBins){
   unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;//mapping to input data

   if (input_id > input_size) {
       return;
   }

   //bin = (lum[i] - lumMin) / lumRange * numBins; calculating the bin to which
   unsigned int bin;
   bin = ((d_input_array[input_id] - lumMin)/lumRange)*numBins;
   if (bin == numBins) {// only in case for pixels with maximum intensity
       bin = numBins -1;
   }
   
   unsigned int tagged;
   volatile unsigned int bin_t = (unsigned int) bin;

   do{
      unsigned int w_val = d_histo_out[bin_t] & 0x7FFFFFF; // using first 5 bits for tagging with thread_id;
      tagged = (input_id << 27) | (w_val+1);
      d_histo_out[bin_t] = tagged;
   } while (d_histo_out[bin_t] != tagged);
   //atomicAdd(&d_histo_out[bin],1);
}

//Hillis-Steele Scan because the array_size is small. So we want Step-Efficient algorithm.
//But, there is an extra step, as we need to do exclusive scan. 
//So, we can be clever in copying data in shared_memory, copying data from one index back and perform inclusive scan
__global__
void hillis_steele_exclusive_scan(unsigned int* const d_out, unsigned int* d_input_array, unsigned int numBins){
   unsigned int bin_id = threadIdx.x;
   
   extern __shared__ unsigned int sh_data_array[];

   if (bin_id == 0){
       sh_data_array[bin_id] = 0;
   } else if (bin_id > numBins){
       sh_data_array[bin_id] = 0;
   } else {
       sh_data_array[bin_id] = d_input_array[bin_id-1]; //exclusive scan
   }

   __syncthreads();

   for (int offset = 1; offset<numBins; offset<<=1){
       int temp;
       if (bin_id >= offset){
           temp = sh_data_array[bin_id - offset] + sh_data_array[bin_id];
       } else {
           temp = sh_data_array[bin_id];
       }
       __syncthreads();
       sh_data_array[bin_id] = temp;
       __syncthreads();
   }
   d_out[bin_id] = sh_data_array[bin_id];
}

//Double-buffered version of hillis-steele scan
//Essentially we have a memory buffer to avoid syncing threads for every read and write.
__global__
void scan_hillis_steele_double_buffered(unsigned int* const d_out, unsigned int* d_input_array, unsigned int numBins){
   unsigned int bin_id = threadIdx.x;
   
   extern __shared__ unsigned int sh_data_array[];

   //Exclusive Scan requires a shifting of data one index ahead.
   if (bin_id == 0){
       sh_data_array[bin_id] = 0;
   } else if (bin_id > numBins){
       sh_data_array[bin_id] = 0;
   } else {
       sh_data_array[bin_id] = d_input_array[bin_id-1]; //exclusive scan
   }

   __syncthreads();

   int t_out=1, t_in=0;//initially, we keep the output in the buffer array, then we keep alternating.
   
   for (int offset =1; offset<numBins; offset<<=1){
       t_in=1-t_out;
       if (bin_id >=offset){
          sh_data_array[t_out*numBins + bin_id] = sh_data_array[t_in*numBins + bin_id - offset] + sh_data_array[t_in*numBins + bin_id];
       } else {
          sh_data_array[t_out*numBins + bin_id] = sh_data_array[t_in*numBins + bin_id];
       }
       t_out = 1-t_out;
       __syncthreads();
   }
   t_out = 1-t_out;
   d_out[bin_id] = sh_data_array[t_out*numBins + bin_id];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
   // Find Min and Max

   int input_size = numRows * numCols;

   min_logLum = min_max_reduce(d_logLuminance, input_size, true);
   max_logLum = min_max_reduce(d_logLuminance, input_size, false);

   float lumRange = max_logLum - min_logLum;

   //Generating Histogram
   unsigned int* d_histo_out;
   unsigned int n_blocks = ceil(1.0f*input_size/BLOCK_SIZE);
   unsigned int SH_MEM_BYTES = sizeof(unsigned int) * numBins;

// SOME EXPERIMENTS

//   unsigned int *h_histo_out_temp = (unsigned int *) malloc(sizeof(unsigned int)*numBins);

  
   checkCudaErrors(cudaMalloc(&d_histo_out, sizeof(unsigned int) * numBins));
   checkCudaErrors(cudaMemset(d_histo_out, 0, sizeof(unsigned int) * numBins));
   

   histogram_atomic_shared<<<n_blocks, BLOCK_SIZE, SH_MEM_BYTES>>>(d_histo_out, d_logLuminance, input_size, min_logLum, lumRange, numBins);

   //checkCudaErrors(cudaMemcpy(h_histo_out_temp, d_histo_out, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

   //std::cout<<"\n";
   //for (int i=0 ; i< numBins; i++)
   //	std::cout<<h_histo_out_temp[i]<<" ";
   //std::cout<<"\n";

   //checkCudaErrors(cudaMemset(d_histo_out, 0, sizeof(unsigned int) * numBins));
   //histogram_atomic<<<n_blocks, BLOCK_SIZE>>>(d_histo_out, d_logLuminance, input_size, min_logLum, lumRange, numBins);

   //checkCudaErrors(cudaMemcpy(h_histo_out_temp, d_histo_out, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

   
   //std::cout<<"\n";
   //for (int i=0 ; i< numBins; i++)
   //	std::cout<<h_histo_out_temp[i]<<" ";
   //std::cout<<"\n";


// EXPERIMENTS END

   //Histogram without atomicAdds.
   //histogram_simulated_atomic<<<n_blocks, BLOCK_SIZE>>>(d_histo_out, d_logLuminance, input_size, min_logLum, lumRange, numBins);

   unsigned int* h_histo_out;
   h_histo_out = (unsigned int*) malloc(sizeof(unsigned int) * numBins);
   checkCudaErrors(cudaMemcpy(h_histo_out, d_histo_out, SH_MEM_BYTES, cudaMemcpyDeviceToHost));

   //std::cout<<"TEST PRINT : "<<h_histo_out[34]<<"\n";
   //std::cout<<"Input Size : "<<input_size<<" Bins : "<<numBins<<"\n";

   checkCudaErrors(cudaGetLastError());

   SH_MEM_BYTES = sizeof(unsigned int) * numBins;

   //Generating CDF - Naive Hillis-Steele Scan
   //hillis_steele_exclusive_scan<<<1, numBins, SH_MEM_BYTES>>>(d_cdf, d_histo_out, numBins);

   //Hillis-Steele Scan with Double Buffered Version
   scan_hillis_steele_double_buffered<<<1, numBins, 2*SH_MEM_BYTES>>>(d_cdf, d_histo_out, numBins);
   checkCudaErrors(cudaGetLastError());
}
