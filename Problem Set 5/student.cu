/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

const int BLOCK_SIZE = 1024;

__global__
void yourNaiveHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code

  unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;//mapping to input data

   if (input_id >= numVals) {
       return;
   }
   
   atomicAdd(&histo[vals[input_id]], 1);

}


__global__
void yourSharedHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals,
               int numBins)
{
  unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;//mapping to input data

   extern __shared__ unsigned int sh_data_array[];

   if (input_id >= numVals) {
       return;
   }
   
   // Make this initialization generic with respect to BLOCK_SIZE
   int num_data_block = numBins/BLOCK_SIZE;
   
   if (num_data_block*BLOCK_SIZE != numBins) {
       num_data_block = num_data_block +1;
   }
   
   for (int i=0; i< num_data_block; i++){
       unsigned int sh_data_index = threadIdx.x + i*BLOCK_SIZE;//mapping to input data
	   if (sh_data_index < numBins){
	       sh_data_array[sh_data_index] = 0;
	   }
   }

   __syncthreads();

   //bin = (lum[i] - lumMin) / lumRange * numBins; calculating the bin to which
  //  unsigned int bin = ((d_input_array[input_id] - lumMin)/lumRange)*numBins;
  //  if (bin == numBins) {// only in case for pixels with maximum intensity
  //      bin = numBins -1;
  //  }
   
   atomicAdd(&sh_data_array[vals[input_id]],1);
   
   __syncthreads();
   
   for (int i=0; i< num_data_block; i++){
       unsigned int sh_data_index = threadIdx.x + i*BLOCK_SIZE;//mapping to input data
	   if (sh_data_index < numBins){
	       atomicAdd(&histo[sh_data_index], sh_data_array[sh_data_index]);
	   }
   }
   
   //if (threadIdx.x < numBins){
   //   atomicAdd(&d_histo_out[threadIdx.x], sh_data_array[threadIdx.x]);
   //}
   
   __syncthreads();

}


void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel
  unsigned int n_blocks = ceil(1.0f*numElems/BLOCK_SIZE);
  
  // checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

  // yourNaiveHisto<<<n_blocks, BLOCK_SIZE>>>(d_vals, d_histo, numElems);
  yourSharedHisto<<<n_blocks, BLOCK_SIZE, sizeof(unsigned int) * numBins>>>(d_vals, d_histo, numElems, numBins);

  //if you want to use/launch more than one kernel,
  //feel free

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
