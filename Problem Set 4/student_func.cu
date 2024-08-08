//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <stdlib.h>
#include <thrust/host_vector.h>

using namespace std;

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

const int BLOCK_SIZE=1024;

////////// DIGIT WISE HISTOGRAM FUNCTION //////////////////
__global__
void histogram_atomic_shared(unsigned int* const d_histo_out, 
							 unsigned int* const d_inputVals,
							 unsigned int input_size,
							 unsigned int mask,
							 unsigned int bit_pos,
							 unsigned int numBins)
{
   //unsigned int input_size = numElems;
   
   unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;//mapping to input data

   extern __shared__ unsigned int sh_data_array[];

   if (input_id >= input_size) {
       return;
   }
   
   // Make this initialization generic with respect to BLOCK_SIZE
   
   unsigned int num_data_block = numBins/BLOCK_SIZE; 
   if (num_data_block*BLOCK_SIZE!=numBins) {
       num_data_block = num_data_block +1;
   }
   
   // Initializing block-wise histogram with zero
   for (int i=0; i< num_data_block; i++){
       unsigned int sh_data_index = threadIdx.x + i*BLOCK_SIZE;//mapping to input data
	   if (sh_data_index < numBins){
	       sh_data_array[sh_data_index] = 0;
	   }
   }

   __syncthreads();

   //unsigned int bin = ((d_input_array[input_id] - lumMin)/lumRange)*numBins;
   unsigned int bin = (d_inputVals[input_id] & mask) >> bit_pos;

   // Updating block-wise histogram appropriately
   atomicAdd(&sh_data_array[bin],1);
   
   __syncthreads();
   
   // Updating global histogram appropriately
   for (int i=0; i< num_data_block; i++){
       unsigned int sh_data_index = threadIdx.x + i*BLOCK_SIZE;//mapping to input data
	   if (sh_data_index < numBins){
	       atomicAdd(&d_histo_out[sh_data_index], sh_data_array[sh_data_index]);
	   }
   }
   
   __syncthreads();
}
////////// DIGIT WISE HISTOGRAM FUNCTION //////////////////



/// PREDICATE FUNCTION ///
__global__ void predicate(unsigned int* d_predicate, 
						const unsigned int* d_in, 
						unsigned int input_size,
						unsigned int mask,
						unsigned int bit_pos,
						unsigned int digit)
{
	// 
	unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;
	// unsigned int tid = threadIdx.x;

	if (input_id >= input_size) {
    	return;
   	}

	unsigned int bin = (d_in[input_id] & mask) >> bit_pos;

	d_predicate[input_id] = (bin == digit) ? 1 : 0;

}

__global__ void get_scatter_address_per_digit(unsigned int* d_scatter_address, 
										const unsigned int* d_scan_per_digit,
										unsigned int* d_predicate,
										unsigned int input_size,
										unsigned int offset)
{
	unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;
	// unsigned int tid = threadIdx.x;

	if (input_id >= input_size) {
    	return;
   	}

	if (d_predicate[input_id] == 1) {
		d_scatter_address[input_id] = offset + d_scan_per_digit[input_id];
	}

}

__global__ void newMoveElements(unsigned int* d_out, 
							unsigned int* d_in,
							const unsigned int* d_scan_per_digit, 
							unsigned int* d_predicate, 
							unsigned int input_size,
							unsigned int offset)
{
	unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;

	if (input_id >= input_size) {
    	return;
   	}

	if (d_predicate[input_id] == 1) {
		unsigned int newIndex = offset + d_scan_per_digit[input_id];
		d_out[newIndex] = d_in[input_id];
	}

}

__global__ void moveElements(unsigned int* d_out, 
							unsigned int* d_in, 
							const unsigned int* d_scatter_address, 
							unsigned int input_size)
{
	unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;
	// unsigned int tid = threadIdx.x;

	if (input_id >= input_size) {
		return;
   	}

	unsigned int newindex = d_scatter_address[input_id];
	// if (newindex >= input_size){
	// 	return;
	// }
	d_out[newindex] = d_in[input_id];
}

__global__ void adjust_block_wise(unsigned int* d_scan, 
								unsigned int* d_block_scan, 
								unsigned int input_size)
{
	unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;
	// unsigned int tid = threadIdx.x;

	if (input_id >= input_size) {
    	return;
   	}

	d_scan[input_id] += d_block_scan[blockIdx.x];
}

__global__ void blelloch_scan_kernel(unsigned int* d_scan, 
									unsigned int* d_predicate, 
									unsigned int* d_blockScanSums,
									unsigned int input_size)
{
	// do a per-block exclusive scan; and add last sum value in d_blockScanSums
	unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tid = threadIdx.x;

	extern __shared__ unsigned int sh_data_array[];

	sh_data_array[tid] = (input_id < input_size) ? d_predicate[input_id] : 0;

	unsigned int offset = 1;
	for (int gap=blockDim.x/2; gap >0; gap >>= 1){
		__syncthreads();
		if (tid < gap) {
			int idx1 = offset*(2*tid +1) - 1;
			int idx2 = offset*(2*tid +2) - 1;
			sh_data_array[idx2] += sh_data_array[idx1];
		}
		offset <<=1;
	}

	// making the last element zero;
	if (tid == 0)
		sh_data_array[blockDim.x - 1] = 0;

	// downsweep
	for (int gap=1; gap <blockDim.x; gap <<= 1){
		offset >>=1;
		__syncthreads();
		if (tid < gap) {
			int idx1 = offset*(2*tid +1) - 1;
			int idx2 = offset*(2*tid +2) - 1;
			unsigned int t = sh_data_array[idx1]; // saving left operand
			sh_data_array[idx1] = sh_data_array[idx2]; // L = R
			sh_data_array[idx2] += t; // R = L+R
		}
	}
	__syncthreads();

	if (input_id < input_size){
		d_scan[input_id] = sh_data_array[tid];
	}

	if (tid==0){
		// copying the final sum in blockarray
		d_blockScanSums[blockIdx.x] = sh_data_array[blockDim.x -1];
		unsigned int last_elem_block = blockDim.x * blockIdx.x + blockDim.x -1 ;
		// also adding the last element of this particular block.
		if (last_elem_block < input_size){
			d_blockScanSums[blockIdx.x] += d_predicate[last_elem_block];
		}
	}
}



unsigned int blellochScan(unsigned int* d_scan, unsigned int* d_predicate, size_t numElems) {
	// the main function to write here
	int num_kernel_blocks = ceil(1.0*numElems/BLOCK_SIZE);
	// cout<<"inside BlellochScan - Current problem size : "<<numElems<<endl;
	unsigned int* d_blockScanSums;

	checkCudaErrors(cudaMalloc(&d_blockScanSums, num_kernel_blocks*sizeof(unsigned int*)));
	blelloch_scan_kernel<<<num_kernel_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(unsigned int)>>>(d_scan, d_predicate, d_blockScanSums, numElems);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());		

	unsigned int total_sum;

	if (num_kernel_blocks > 1 ){
		// we would need to do a recusion block-wise;
		unsigned int* d_scan_temp;
		checkCudaErrors(cudaMalloc(&d_scan_temp, num_kernel_blocks*sizeof(unsigned int*)));
		total_sum = blellochScan(d_scan_temp, d_blockScanSums, num_kernel_blocks);
		// we need to now adjust every block with blocksums from d_scan_temp;
		adjust_block_wise<<<num_kernel_blocks, BLOCK_SIZE>>>(d_scan, d_scan_temp, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());		
		checkCudaErrors(cudaFree(d_scan_temp));
	} else {
		checkCudaErrors(cudaMemcpy(&total_sum, d_blockScanSums, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_blockScanSums));
	}

	return total_sum ; 
}



// HELPER FUNCTIONS
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


void print_help(unsigned int* d_outputVals, unsigned int* d_outputPos, const size_t numElems){
	unsigned int* h_outputVals;
	unsigned int* h_outputPos;

	h_outputVals = (unsigned int *) malloc(numElems * sizeof(unsigned int *));
	h_outputPos = (unsigned int *) malloc(numElems * sizeof(unsigned int *));

	// checkCudaErrors(cudaMalloc(&d_histo_out, sizeof(unsigned int) * numElems));
	// checkCudaErrors(cudaMalloc(&d_histo_out, sizeof(unsigned int) * numElems));

	checkCudaErrors(cudaMemcpy(h_outputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_outputPos, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	for (int i=0; i< numElems; i+=1000){
		cout<<"Output Pos : \t"<<h_outputPos[i]<<"\t\t Output Val : \t"<<h_outputVals[i]<<endl;
	}
	return;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE

  //2. I need the scatter address ALLOCATE going from least significant bit to most significant bit. Loop that many times
		// a. For the scatter address, I need to get the scan of true values.
		// b. True values I would get by finding the digit at the current bit place and making a predicate array.
		// c. Using the predicate, need to do a scan on that for each digit separately
		// d. For digit "0", this scan would be sufficient, but for other digits would need an offset for them.
		// e. This offset can be precalculated using digit histograms. Histograms can be calculated independently of sort and kept handy even before the actual sort begins.
		// f. We can avoid sorting where a particular position has only one digit. 
  
	const int numBits = 1;
	const int numBins = 1 << numBits;
	
	const int bitBlocks = 8 * sizeof(unsigned int) / numBits;
	
	//unsigned int **binHistogram = new unsigned int[bitBlocks][numBins];
    //unsigned int **binScan      = new unsigned int[bitBlocks][numBins];
	
	unsigned int **binHistogram;
    unsigned int **binScan;
	
	// binHistogram = (unsigned int **) malloc(bitBlocks * sizeof(unsigned int *));
	// binScan = (unsigned int **) malloc(bitBlocks * sizeof(unsigned int *));
	
	// for (int i=0; i<bitBlocks; i++){
	// 	binHistogram[i] = new unsigned int[numBins];
	// 	binScan[i] = new unsigned int[numBins];
	// 	memset(binHistogram[i], 0, sizeof(unsigned int) * numBins); //zero out the bins
	// 	memset(binScan[i], 0, sizeof(unsigned int) * numBins ); //zero out the bins
	// }
	
	//memset(binHistogram, 0, sizeof(unsigned int) * numBins * bitBlocks); //zero out the bins
    //memset(binScan, 0, sizeof(unsigned int) * numBins * bitBlocks); //zero out the bins
	
	// cout<<"INITIALIZING HIST "<<endl;
	// for (int i=0; i< bitBlocks; i++){
	// 	cout<<endl;
	// 	cout<<"BITBLOCK: "<<i<<" HIST: ";
	// 	for (int j=0; j < numBins; j++){
	// 		cout<<binHistogram[i][j]<<" ";	
	// 	}
	// 	cout<<" SCAN: ";
	// 	for (int j=0; j < numBins; j++){
	// 		cout<<binScan[i][j]<<" ";	
	// 	}
	// }
	
	cout<<endl;
	
	//setting temporary pointers to input and output variables
	
	unsigned int *vals_src = d_inputVals;
    unsigned int *pos_src  = d_inputPos;

    unsigned int *vals_dst = d_outputVals;
    unsigned int *pos_dst  = d_outputPos;
	
	// AFTER CAREFUL CONSIDERATION, PHASE ONE NOT NEEDED NOW, SINCE DIGIT_COUNT SCAN CAN BE DONE IN THE SAME LOOP AS MOVING
	// PHASE ONE :- CALCULATE HISTOGRAM FOR EVERY BIT_BLOCK
	// 32 rows if for all bits... and 16 if say for 2-bit blocks
	unsigned int* d_histo_out;
	checkCudaErrors(cudaMalloc(&d_histo_out, sizeof(unsigned int) * numBins));
	
	// for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
		
	// 	checkCudaErrors(cudaMemset(d_histo_out, 0, sizeof(unsigned int) * numBins));
		
	// 	unsigned int mask = (numBins - 1) << i;
		
		
	// 	unsigned int n_blocks = ceil(1.0f*numElems/BLOCK_SIZE);
	// 	unsigned int SH_MEM_BYTES = sizeof(unsigned int) * numBins;	
		
		
	// 	histogram_atomic_shared<<<n_blocks, BLOCK_SIZE, SH_MEM_BYTES>>>(d_histo_out, d_inputVals, numElems, mask, i, numBins);
		
	// 	//cout<<"REACHING HERE?? "<<i<<" "<<endl;
		
	// 	unsigned int bit_block = i/numBits;
	// 	checkCudaErrors(cudaMemcpy(binHistogram[bit_block], d_histo_out, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));
		
	// 	// by now histogram of this group of digits must be at binHistogram[bit_block] position;
		
	// 	// exclusive sum-scan on digit histogram
	// 	for (unsigned int j = 1; j < numBins; ++j) {
	// 		binScan[bit_block][j] = binScan[bit_block][j - 1] + binHistogram[bit_block][j - 1];
	// 	}
		
	// 	// if logic and syntax is correct, I have now every position's Offset position correctly calculated.

	// }
	
	//
	//cout<<"IS THIS COMPLETED?"<<endl;
	
	//for (int i=0; i< bitBlocks; i++){
	//	cout<<endl;
	//	cout<<"BITBLOCK: "<<i<<" HIST: ";
	//	for (int j=0; j < numBins; j++){
	//		cout<<binHistogram[i][j]<<" ";	
	//	}
	//	cout<<" SCAN: ";
	//	for (int j=0; j < numBins; j++){
	//		cout<<binScan[i][j]<<" ";	
	//	}
	//}	
	cout<<endl;

	// unsigned int* d_histo;
	unsigned int* d_predicate;
	unsigned int* d_scan_per_digit;
	unsigned int* d_scatter_address;

	checkCudaErrors(cudaMalloc(&d_predicate, numElems*sizeof(unsigned int*)));
	checkCudaErrors(cudaMalloc(&d_scan_per_digit, numElems*sizeof(unsigned int*)));
	checkCudaErrors(cudaMalloc(&d_scatter_address, numElems*sizeof(unsigned int*)));

	int num_kernel_blocks = ceil(1.0*numElems/BLOCK_SIZE);

	// PHASE TWO:- ACTUAL SCAN & SORTING
	for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
		// In every loop, bit_block is fixed
		unsigned int bit_block = i/numBits;
		
		unsigned int mask = (numBins - 1) << i;
		
		// cout<<"REFERENCE CALCULATION FOR BITBLOCK: "<<i<<" "<<endl;

		unsigned int offset = 0;

		for (int digit=0; digit<numBins; digit++){
			// Now digit is fixed
			// unsigned int offset = binScan[bit_block][digit];
			// cout<<"COUNT for DIGIT :"<<digit<<" is :"<<binHistogram[bit_block][digit]<<endl;
			// cout<<"OFFSET for DIGIT :"<<digit<<" is :"<<offset<<endl;
			// calculate predicate digit-wise
			checkCudaErrors(cudaMemset(d_scan_per_digit, 0, sizeof(unsigned int) * numElems));
			checkCudaErrors(cudaMemset(d_predicate, 0, sizeof(unsigned int) * numElems));
			predicate<<<num_kernel_blocks, BLOCK_SIZE>>>(d_predicate, vals_src, numElems, mask, i, digit);
			// do a digit-wise scan, i.e. predicate for rest of digits is zero;
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			
			// NOW ONLY SCAN NEED TO BE DONE CORRECTLY

			unsigned int total_sum = blellochScan(d_scan_per_digit, d_predicate, numElems);
			
			// cout<<"TOTAL_SUM for DIGIT :"<<digit<<" is :"<<total_sum<<endl;
			
			// if (digit == 1){
			// 	unsigned int* h_scan_per_digit;
			// 	h_scan_per_digit = (unsigned int *) malloc(numElems * sizeof(unsigned int *));
			// 	checkCudaErrors(cudaMemcpy(h_scan_per_digit, d_scan_per_digit, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
			// 	cout<<"Output Scanned Address : "<<endl;
			// 	for (int i=0; i< numElems; i+=1000){
			// 		cout<<h_scan_per_digit[i]<<" ";
			// 	}
			// 	cout<<endl;
			// }
				

			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			//add offset address from histogram digit-wise and store it in d_scatter_address;
			// get_scatter_address_per_digit<<<num_kernel_blocks, BLOCK_SIZE>>>(d_scatter_address, d_scan_per_digit, d_predicate, numElems, offset);

			newMoveElements<<<num_kernel_blocks, BLOCK_SIZE>>>(vals_dst, vals_src, d_scan_per_digit, d_predicate, numElems, offset);
			newMoveElements<<<num_kernel_blocks, BLOCK_SIZE>>>(pos_dst, pos_src, d_scan_per_digit, d_predicate, numElems, offset);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			offset += total_sum;


		}
		
		// cout<<"MOVING ELEMENTS for BITBLOCK: "<<i<<" "<<endl;
		// moveElements<<<num_kernel_blocks, BLOCK_SIZE>>>(vals_dst, vals_src, d_scatter_address, numElems);
		// moveElements<<<num_kernel_blocks, BLOCK_SIZE>>>(pos_dst, pos_src, d_scatter_address, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		// cout<<"ELEMENTS MOVED for BITBLOCK: "<<i<<" "<<endl;
	
		//swap the buffers (pointers only)
		std::swap(vals_dst, vals_src);
		std::swap(pos_dst, pos_src);
	
	}

	// cout<<"REACHING THE END OF PHASE TWO. "<<endl;

	checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

	// print_help(d_outputVals, d_outputPos, numElems); // ADDED BY ANKIT
	
	checkCudaErrors(cudaFree(d_predicate));
	checkCudaErrors(cudaFree(d_scan_per_digit));
	checkCudaErrors(cudaFree(d_scatter_address));
	//we did an even number of iterations, need to copy from input buffer into output
  
}
