//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
	  as boundary conditions for solving a Poisson equation that tells
	  us how to blend the images.
   
	  No pixels from the destination except pixels on the border
	  are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
	  Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
			 else if the neighbor in on the border then += DestinationImg[neighbor]

	  Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
	  float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
	  ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


	In this assignment we will do 800 iterations.
   */
#include "utils.h"
#include <stdio.h>
#include <vector>
#include "device_launch_parameters.h"

const int BLOCK_WIDTH = 16;
const int filterWidth = 3;

__device__
int clamp(int a, int lo, int hi){
   int ret;
   ret = lo <= a ? a : lo;
   ret = ret < hi ? ret : hi -1;
   return ret;
}


__device__ bool insideMask(uchar4 val) {
	return !(val.x == 255 && val.y == 255 && val.z == 255);
}


__global__
void separateChannels(const uchar4* const inputImageRGBA,
					  int numRows,
					  int numCols,
					  float* const redChannel,
					  float* const greenChannel,
					  float* const blueChannel)
{
   // TODO
   //
   // NOTE: Be careful not to try to access memory that is outside the bounds of
   // the image. You'll want code that performs the following check before accessing
   // GPU memory:
   //
   // if ( absolute_image_position_x >= numCols ||
   //      absolute_image_position_y >= numRows )
   // {
   //     return;
   // }
   unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
   unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;

   if (row >= numRows || col >=numCols){
	  return;
   }
   
   unsigned int image1DIndex = numCols * row + col;

   redChannel[image1DIndex] = (float) inputImageRGBA[image1DIndex].x;
   greenChannel[image1DIndex] = (float) inputImageRGBA[image1DIndex].y;
   blueChannel[image1DIndex] = (float) inputImageRGBA[image1DIndex].z;

}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const uchar4* const d_destImg,
					   unsigned char* interior,
					   const float* const redChannel,
					   const float* const greenChannel,
					   const float* const blueChannel,
					   uchar4* const outputImageRGBA,
					   int numRows,
					   int numCols)
{
   const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
										blockIdx.y * blockDim.y + threadIdx.y);

   const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

   //make sure we don't try and access memory outside the image
   //by having any threads mapped there return early
   if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
	  return;

   if (interior[thread_1D_pos] != 1){
	  outputImageRGBA[thread_1D_pos] = d_destImg[thread_1D_pos];
	  return;
   }

   unsigned char red   = (unsigned char)redChannel[thread_1D_pos];
   unsigned char green = (unsigned char)greenChannel[thread_1D_pos];
   unsigned char blue  = (unsigned char)blueChannel[thread_1D_pos];

   //Alpha should be 255 for no transparency
   uchar4 outputPixel = make_uchar4(red, green, blue, 255);

   outputImageRGBA[thread_1D_pos] = outputPixel;
}

__device__
float min_k(float a, float b){
	if (a < 0)a = 999999;
	if (b < 0)b = 999999;
   	return a < b ? a : b;
}

__device__
float max_k(float a, float b){
   	return a > b ? a : b;
}

//KERNEL to reduce in a block and storing the result at the blockIdx location in global memory in d_out array
//Since, max_reduce would be almost similar, we incorporate a bool to make the switch in the associative operator, the writing back of result part.
__global__
void min_max_reduce(int* d_out , const int* const d_in, int input_size, bool isMin){

   unsigned int input_id = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int tid = threadIdx.x;

   extern __shared__ int sh_data[];

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
int min_max_reduce(const int* const d_logLuminance, int input_size, bool isMin){
   int problem_size = input_size;
   int *d_intermediate = NULL;
   int BLOCK_SIZE = BLOCK_WIDTH;

   unsigned int n_blocks = ceil(1.0f*problem_size/BLOCK_SIZE);
   unsigned int threads = BLOCK_SIZE;
   unsigned int SH_MEM_BYTES = threads * sizeof(int);

   while (true) {
      int* d_blockMin = NULL; //For Storing Intermediate results for a block
      checkCudaErrors(cudaMalloc(&d_blockMin,   sizeof(int) * n_blocks));

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
         int h_min;
         //Only a single block was launched. d_blockMin[0] should have answer.
         //cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
         checkCudaErrors(cudaMemcpy(&h_min, d_intermediate, sizeof(int), cudaMemcpyDeviceToHost));
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
         SH_MEM_BYTES = threads * sizeof(int);
      }
   }

}


// Compute the Mask
__global__
void get_borders_and_interiors(const uchar4* const d_sourceImg, 
							  unsigned char *border, 
							  unsigned char *interior,
							  int* xcoords,
							  int* ycoords,
							  const size_t numRows, 
							  const size_t numCols)
{
   
   int outputRow = threadIdx.y + blockIdx.y * blockDim.y;
   int outputCol = threadIdx.x + blockIdx.x * blockDim.x;

   unsigned int image1DIndexOutput = outputRow * numCols + outputCol;

   __shared__ uchar4 sh_img_block[(BLOCK_WIDTH + filterWidth - 1)*(BLOCK_WIDTH + filterWidth - 1)];

   //shared memory is a block of square of size = blocksize + filterWidth -1;
   int sh_mem_block_size = BLOCK_WIDTH + filterWidth-1;
   int halfWidth = filterWidth/2;

   int ori_image_row_pos = outputRow - halfWidth;
   int ori_image_col_pos = outputCol - halfWidth;

   int sh_img_row_pos = threadIdx.y;
   int sh_img_col_pos = threadIdx.x;

   ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
   ori_image_col_pos = clamp(ori_image_col_pos, 0, numCols);

   /// LOAD DATA IN SHARED MEMORY ///

   // current pixel mapped to shared_memory location!
   sh_img_block[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos] = d_sourceImg[ori_image_row_pos * numCols + ori_image_col_pos];

   int boundary_thresh_row = blockDim.y - filterWidth+1;
   int boundary_thresh_col = blockDim.x - filterWidth+1;

   if (threadIdx.y >= boundary_thresh_row){
	  // filling in the lower bottom rectangle of size (filterWidth -1 ) * (blockDim.x) of shared memory except for bottom-right corner
	  ori_image_row_pos = outputRow + halfWidth;
	  ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
	  sh_img_block[(sh_img_row_pos + filterWidth -1)* sh_mem_block_size + sh_img_col_pos] = d_sourceImg[ori_image_row_pos * numCols + ori_image_col_pos];
   }

   if (threadIdx.x >= boundary_thresh_col){
	  // filling in the right left rectangle of size (blockDim.y) * (filterWidth - 1) of shared memory except for bottom-right corner
	  ori_image_row_pos = outputRow - halfWidth;
	  ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
	  ori_image_col_pos = outputCol + halfWidth;
	  ori_image_col_pos = clamp(ori_image_col_pos, 0, numCols);
	  sh_img_block[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos + filterWidth-1] = d_sourceImg[ori_image_row_pos * numCols + ori_image_col_pos];
   }
   
   if (threadIdx.x < (filterWidth - 1) && threadIdx.y < (filterWidth-1)) {
	  // filling the bottom-right corner of zie (filterWidth - 1) ^ (2).
	  ori_image_row_pos = outputRow - halfWidth + blockDim.y;
	  ori_image_col_pos = outputCol - halfWidth + blockDim.x;

	  ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
	  ori_image_col_pos = clamp(ori_image_col_pos, 0, numCols);

	  sh_img_block[(sh_img_row_pos + blockDim.y)*sh_mem_block_size + sh_img_col_pos + blockDim.x] = d_sourceImg[ori_image_row_pos * numCols + ori_image_col_pos];
   }

   __syncthreads();
   /// DATA LOADED IN SHARED MEMORY ///

   if (outputRow >= numRows || outputCol >=numCols){
	  return; // out of image boundary. nothing to do
   }

   int sh_img_target_row_pos = threadIdx.y + halfWidth;
   int sh_img_target_col_pos = threadIdx.x + halfWidth;   

   if (!insideMask(sh_img_block[sh_img_target_row_pos*sh_mem_block_size + sh_img_target_col_pos])) {
	  return ; // current 2D index is not inside mask
   }

   if (!insideMask(sh_img_block[sh_img_target_row_pos*sh_mem_block_size + sh_img_target_col_pos - 1]) ||
	  !insideMask(sh_img_block[sh_img_target_row_pos*sh_mem_block_size + sh_img_target_col_pos + 1]) ||
	  !insideMask(sh_img_block[(sh_img_target_row_pos - 1)*sh_mem_block_size + sh_img_target_col_pos]) ||
	  !insideMask(sh_img_block[(sh_img_target_row_pos + 1)*sh_mem_block_size + sh_img_target_col_pos]))
   {
	  // Border Pixel
	  border[image1DIndexOutput] = 1;
	  xcoords[image1DIndexOutput] = outputCol;
	  ycoords[image1DIndexOutput] = outputRow;
	  return;
   }

   // Interior Pixel
   interior[image1DIndexOutput] = 1;

}

__global__
void jacobi(float* const d_dest, 
			float* const d_src, 
			unsigned char *border, 
			unsigned char *interior, 
			float* const d_prev, 
			float* d_next, 
			int minx, 
			int miny, 
			int numRows, 
			int numCols)
{
   int outputRow = miny + threadIdx.y + blockIdx.y * blockDim.y;
   int outputCol = minx + threadIdx.x + blockIdx.x * blockDim.x;

   unsigned int image1DIndexOutput = outputRow * numCols + outputCol;

   // ALWAYS ASSUMING SQUARE BLOCKS
   __shared__ float sh_dest[(BLOCK_WIDTH + filterWidth - 1)*(BLOCK_WIDTH + filterWidth - 1)];
   __shared__ float sh_src[(BLOCK_WIDTH + filterWidth - 1)*(BLOCK_WIDTH + filterWidth - 1)];
   __shared__ float sh_prev[(BLOCK_WIDTH + filterWidth - 1)*(BLOCK_WIDTH + filterWidth - 1)];
   __shared__ unsigned char sh_border[(BLOCK_WIDTH + filterWidth - 1)*(BLOCK_WIDTH + filterWidth - 1)];
   __shared__ unsigned char sh_interior[(BLOCK_WIDTH + filterWidth - 1)*(BLOCK_WIDTH + filterWidth - 1)];

   //shared memory is a block of square of size = blocksize + filterWidth -1;
   int sh_mem_block_size = BLOCK_WIDTH + filterWidth-1;
   int halfWidth = filterWidth/2;

   int ori_image_row_pos = outputRow - halfWidth;
   int ori_image_col_pos = outputCol - halfWidth;

   int sh_img_row_pos = threadIdx.y;
   int sh_img_col_pos = threadIdx.x;

   ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
   ori_image_col_pos = clamp(ori_image_col_pos, 0, numCols);

   /// LOAD DATA IN SHARED MEMORY ///

   // // current pixel mapped to shared_memory location!
   sh_dest[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos] = d_dest[ori_image_row_pos * numCols + ori_image_col_pos];
   sh_src[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos] = d_src[ori_image_row_pos * numCols + ori_image_col_pos];
   sh_prev[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos] = d_prev[ori_image_row_pos * numCols + ori_image_col_pos];

   sh_border[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos] = border[ori_image_row_pos * numCols + ori_image_col_pos];
   sh_interior[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos] = interior[ori_image_row_pos * numCols + ori_image_col_pos];

   int boundary_thresh_row = blockDim.y - filterWidth+1;
   int boundary_thresh_col = blockDim.x - filterWidth+1;

   if (threadIdx.y >= boundary_thresh_row){
	  // filling in the lower bottom rectangle of size (filterWidth -1 ) * (blockDim.x) of shared memory except for bottom-right corner
	  ori_image_row_pos = outputRow + halfWidth;
	  ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
	  sh_dest[(sh_img_row_pos + filterWidth -1)* sh_mem_block_size + sh_img_col_pos] = d_dest[ori_image_row_pos * numCols + ori_image_col_pos];
	  sh_src[(sh_img_row_pos + filterWidth -1)* sh_mem_block_size + sh_img_col_pos] = d_src[ori_image_row_pos * numCols + ori_image_col_pos];
	  sh_prev[(sh_img_row_pos + filterWidth -1)* sh_mem_block_size + sh_img_col_pos] = d_prev[ori_image_row_pos * numCols + ori_image_col_pos];

	  sh_border[(sh_img_row_pos + filterWidth -1)* sh_mem_block_size + sh_img_col_pos] = border[ori_image_row_pos * numCols + ori_image_col_pos];
	  sh_interior[(sh_img_row_pos + filterWidth -1)* sh_mem_block_size + sh_img_col_pos] = interior[ori_image_row_pos * numCols + ori_image_col_pos];
   }

   if (threadIdx.x >= boundary_thresh_col){
	  // filling in the right left rectangle of size (blockDim.y) * (filterWidth - 1) of shared memory except for bottom-right corner
	  ori_image_row_pos = outputRow - halfWidth;
	  ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
	  ori_image_col_pos = outputCol + halfWidth;
	  ori_image_col_pos = clamp(ori_image_col_pos, 0, numCols);
	  sh_dest[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos + filterWidth-1] = d_dest[ori_image_row_pos * numCols + ori_image_col_pos];
	  sh_src[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos + filterWidth-1] = d_src[ori_image_row_pos * numCols + ori_image_col_pos];
	  sh_prev[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos + filterWidth-1] = d_prev[ori_image_row_pos * numCols + ori_image_col_pos];

	  sh_border[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos + filterWidth-1] = border[ori_image_row_pos * numCols + ori_image_col_pos];
	  sh_interior[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos + filterWidth-1] = interior[ori_image_row_pos * numCols + ori_image_col_pos];
   }
   
   if (threadIdx.x < (filterWidth - 1) && threadIdx.y < (filterWidth-1)) {
	  // filling the bottom-right corner of zie (filterWidth - 1) ^ (2).
	  ori_image_row_pos = outputRow - halfWidth + blockDim.y;
	  ori_image_col_pos = outputCol - halfWidth + blockDim.x;

	  ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
	  ori_image_col_pos = clamp(ori_image_col_pos, 0, numCols);

	  sh_dest[(sh_img_row_pos + blockDim.y)*sh_mem_block_size + sh_img_col_pos + blockDim.x] = d_dest[ori_image_row_pos * numCols + ori_image_col_pos];
	  sh_src[(sh_img_row_pos + blockDim.y)*sh_mem_block_size + sh_img_col_pos + blockDim.x] = d_src[ori_image_row_pos * numCols + ori_image_col_pos];
	  sh_prev[(sh_img_row_pos + blockDim.y)*sh_mem_block_size + sh_img_col_pos + blockDim.x] = d_prev[ori_image_row_pos * numCols + ori_image_col_pos];

	  sh_border[(sh_img_row_pos + blockDim.y)*sh_mem_block_size + sh_img_col_pos + blockDim.x] = border[ori_image_row_pos * numCols + ori_image_col_pos];
	  sh_interior[(sh_img_row_pos + blockDim.y)*sh_mem_block_size + sh_img_col_pos + blockDim.x] = interior[ori_image_row_pos * numCols + ori_image_col_pos];
   }

   __syncthreads();
   /// DATA LOADED IN SHARED MEMORY ///
   
   if (outputRow >= numRows || outputCol >=numCols){
	  return; // out of image boundary. nothing to do
   }

   if (interior[image1DIndexOutput] != 1) return; //filter out boundary points

   int sh_img_target_row_pos = threadIdx.y + halfWidth;
   int sh_img_target_col_pos = threadIdx.x + halfWidth;

   float sum1=0.0f;
   float sum2=4 * sh_src[sh_img_target_row_pos*sh_mem_block_size + sh_img_target_col_pos];

   // Cater to all neighbors, 
   int neighbors[] = {(sh_img_target_row_pos-1)*sh_mem_block_size + sh_img_target_col_pos, 
	  (sh_img_target_row_pos)*sh_mem_block_size + sh_img_target_col_pos-1,
	  (sh_img_target_row_pos+1)*sh_mem_block_size + sh_img_target_col_pos,
	  (sh_img_target_row_pos)*sh_mem_block_size + sh_img_target_col_pos+1};


   for (int i=0; i<4; i++){
	  if (sh_interior[neighbors[i]]){
		 sum1 += sh_prev[neighbors[i]];
	  } else if (sh_border[neighbors[i]]){
		 sum1 += sh_dest[neighbors[i]];
	  }
	  sum2 -= sh_src[neighbors[i]];
   }

   	float newVal = (sum1 + sum2) / 4.0f;
	newVal = newVal < 0 ? 0 : newVal;
	newVal = newVal > 255 ? 255 : newVal;

	d_next[image1DIndexOutput] = newVal;

}


void your_blend(const uchar4* const h_sourceImg,  //IN
				const size_t numRowsSource, const size_t numColsSource,
				const uchar4* const h_destImg, //IN
				uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
  
	 1) Compute a mask of the pixels from the source image to be copied
		The pixels that shouldn't be copied are completely white, they
		have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

	 2) Compute the interior and border regions of the mask.  An interior
		pixel has all 4 neighbors also inside the mask.  A border pixel is
		in the mask itself, but has at least one neighbor that isn't.

	 3) Separate out the incoming image into three separate channels

	 4) Create two float(!) buffers for each color channel that will
		act as our guesses.  Initialize them to the respective color
		channel of the source image since that will act as our intial guess.

	 5) For each color channel perform the Jacobi iteration described 
		above 800 times.

	 6) Create the output image by replacing all the interior pixels
		in the destination image with the result of the Jacobi iterations.
		Just cast the floating point values to unsigned chars since we have
		already made sure to clamp them to the correct range.

	  Since this is final assignment we provide little boilerplate code to
	  help you.  Notice that all the input/output pointers are HOST pointers.

	  You will have to allocate all of your own GPU memory and perform your own
	  memcopies to get data in and out of the GPU memory.

	  Remember to wrap all of your calls with checkCudaErrors() to catch any
	  thing that might go wrong.  After each kernel call do:

	  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	  to catch any errors that happened while executing the kernel.
  */

   const int blockWidth = BLOCK_WIDTH;

   const int blocksX = ceil(1.0f*numColsSource /blockWidth);
   const int blocksY = ceil(1.0f*numRowsSource /blockWidth);

   //TODO: Set reasonable block size (i.e., number of threads per block)
   const dim3 blockSize(blockWidth, blockWidth,1);

   //TODO:
   //Compute correct grid size (i.e., number of blocks per kernel launch)
   //from the image size and and block size.
   const dim3 gridSize(blocksX, blocksY, 1);


   uchar4* d_sourceImg;
   checkCudaErrors(cudaMalloc(&d_sourceImg,   sizeof(uchar4) * numColsSource * numRowsSource));
   checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4)*numColsSource*numRowsSource, cudaMemcpyHostToDevice));


    unsigned char *border,*interior;
	int *xcoords, *ycoords; //for bounding box computation

    checkCudaErrors(cudaMalloc(&border, numRowsSource*numColsSource * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc(&interior, numRowsSource*numColsSource * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc(&xcoords, numRowsSource*numColsSource * sizeof(int)));
	checkCudaErrors(cudaMalloc(&ycoords, numRowsSource*numColsSource * sizeof(int)));
	
	checkCudaErrors(cudaMemset(border, 0, numRowsSource*numColsSource * sizeof(unsigned char)));
	checkCudaErrors(cudaMemset(interior, 0, numRowsSource*numColsSource * sizeof(unsigned char)));
	checkCudaErrors(cudaMemset(xcoords, -1, numRowsSource*numColsSource * sizeof(int)));
	checkCudaErrors(cudaMemset(ycoords, -1, numRowsSource*numColsSource * sizeof(int)));
   

    // Getting the borders and Interiors of the SourceImage
    get_borders_and_interiors<<<gridSize, blockSize>>>(d_sourceImg, border, interior, xcoords, ycoords, numRowsSource, numColsSource);

    uchar4* d_destImg;
	checkCudaErrors(cudaMalloc(&d_destImg, numRowsSource*numColsSource * sizeof(uchar4)));
	checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, numRowsSource*numColsSource * sizeof(uchar4), cudaMemcpyHostToDevice));
   
	
	int minx = min_max_reduce(xcoords, numRowsSource*numColsSource, true);
	int miny = min_max_reduce(ycoords, numRowsSource*numColsSource, true);
	int maxx = min_max_reduce(xcoords, numRowsSource*numColsSource, false);
	int maxy = min_max_reduce(ycoords, numRowsSource*numColsSource, false);

	int size_x = maxx - minx+1;
	int size_y = maxy - miny + 1;

    std::cout<<"MINX MINE"<<std::endl;
    std::cout<<minx<<" "<<miny<<" "<<size_x<<" "<<size_y<<std::endl;

	checkCudaErrors(cudaFree(xcoords));
	checkCudaErrors(cudaFree(ycoords));

   	// int minx=182, miny=131, size_x=206, size_y=92;
//    int minx=0, miny=0, size_x=numColsSource, size_y=numRowsSource;

   	//Preparing to separate the destination image. Point 3 and 4.

   	float *d_red, *d_green, *d_blue;

	checkCudaErrors(cudaMalloc(&d_red,   sizeof(float) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(float) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMalloc(&d_blue,  sizeof(float) * numRowsSource * numColsSource));

	separateChannels << <gridSize, blockSize >> > (d_destImg, numRowsSource, numColsSource, d_red, d_green, d_blue);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   	float *d_buffer_red_1, *d_buffer_red_2;
	float *d_buffer_green_1, *d_buffer_green_2;
	float *d_buffer_blue_1, *d_buffer_blue_2;

   	checkCudaErrors(cudaMalloc(&d_buffer_red_1, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_buffer_red_2, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_buffer_green_1, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_buffer_green_2, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_buffer_blue_1, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_buffer_blue_2, numRowsSource*numColsSource * sizeof(float)));

   	// Separating the source image in separate channels

   	float *d_red_source, *d_green_source, *d_blue_source;

	checkCudaErrors(cudaMalloc(&d_red_source,   sizeof(float)*numRowsSource*numColsSource));
	checkCudaErrors(cudaMalloc(&d_green_source, sizeof(float)*numRowsSource*numColsSource));
	checkCudaErrors(cudaMalloc(&d_blue_source,  sizeof(float)*numRowsSource*numColsSource));
   
   	separateChannels <<<gridSize, blockSize >>> (d_sourceImg, numRowsSource, numColsSource, d_red_source, d_green_source, d_blue_source);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   // First guess is the source image itself.
   cudaStream_t s1, s2, s3;
	cudaStreamCreate(&s1); cudaStreamCreate(&s2); cudaStreamCreate(&s3);

//    checkCudaErrors(cudaMemcpy(d_buffer_red_1, d_red_source, numRowsSource*numColsSource * sizeof(float), cudaMemcpyDeviceToDevice));
//    checkCudaErrors(cudaMemcpy(d_buffer_green_1, d_green_source, numRowsSource*numColsSource * sizeof(float), cudaMemcpyDeviceToDevice));
//    checkCudaErrors(cudaMemcpy(d_buffer_blue_1, d_blue_source, numRowsSource*numColsSource * sizeof(float), cudaMemcpyDeviceToDevice));
   // std::cout<<"HERE "<<std::endl;

	checkCudaErrors(cudaMemcpyAsync(d_buffer_red_1, d_red_source, numRowsSource*numColsSource * sizeof(float), cudaMemcpyDeviceToDevice, s1));
	checkCudaErrors(cudaMemcpyAsync(d_buffer_green_1, d_green_source, numRowsSource*numColsSource * sizeof(float), cudaMemcpyDeviceToDevice, s2));
	checkCudaErrors(cudaMemcpyAsync(d_buffer_blue_1, d_blue_source, numRowsSource*numColsSource * sizeof(float), cudaMemcpyDeviceToDevice, s3));
   
//    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


   // Run the Jacobi iterations as described

//    int minx=0, miny=0;

	const dim3 gridSizeNew(ceil(1.0f*size_x / blockWidth), ceil(1.0f*size_y / blockWidth));


   for (int i=0; i<800; i++){
	  if (i%2==0){
		 // source here is buffer 1
		 jacobi<<<gridSizeNew, blockSize, 0, s1>>> (d_red, d_red_source, border, interior, d_buffer_red_1, d_buffer_red_2,minx,miny,numRowsSource,numColsSource);
		 jacobi<<<gridSizeNew, blockSize, 0, s2>>> (d_green, d_green_source, border, interior, d_buffer_green_1, d_buffer_green_2,minx,miny,numRowsSource,numColsSource);
		 jacobi<<<gridSizeNew, blockSize, 0, s3>>> (d_blue, d_blue_source, border, interior, d_buffer_blue_1, d_buffer_blue_2,minx,miny,numRowsSource,numColsSource);
		 // std::cout<<"Iteration Count : "<<i<<std::endl;
	  } else {
		 // source here is buffer 2
		 jacobi<<<gridSizeNew, blockSize, 0, s1>>> (d_red, d_red_source, border, interior, d_buffer_red_2, d_buffer_red_1,minx,miny,numRowsSource,numColsSource);
		 jacobi<<<gridSizeNew, blockSize, 0, s2>>> (d_green, d_green_source, border, interior, d_buffer_green_2, d_buffer_green_1,minx,miny,numRowsSource,numColsSource);
		 jacobi<<<gridSizeNew, blockSize, 0, s3>>> (d_blue, d_blue_source, border, interior, d_buffer_blue_2, d_buffer_blue_1,minx,miny,numRowsSource,numColsSource);
		 // std::cout<<"Iteration Count : "<<i<<std::endl;
	  }
		 
   }

	// std::cout<<"AFTER JACOBI "<<s1<<std::endl;

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	cudaStreamDestroy(s1); cudaStreamDestroy(s2); cudaStreamDestroy(s3);
 
	std::cout<<"AFTER JACOBI "<<std::endl;

    float* to_print;
	to_print = (float *) malloc(numRowsSource*numColsSource * sizeof(float));
	checkCudaErrors(cudaMemcpy(to_print, d_buffer_green_1, numRowsSource*numColsSource * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout<<"DEBUGGING GREEN : interior "<<numRowsSource*numColsSource<<std::endl;
    for (long unsigned int i=0; i < numRowsSource*numColsSource; i++){
        if (i==86268 || i==86269 || i==86270 || i==86271 || i==86272)
            std::cout<<to_print[i]<<", "<<(int)(to_print[i])<<std::endl;
        // i += 1000;
    }
    std::cout<<std::endl;
    std::cout<<std::endl;

   	
   

   	uchar4* d_blendedImg;
	checkCudaErrors(cudaMalloc(&d_blendedImg, numRowsSource*numColsSource * sizeof(uchar4)));
	recombineChannels <<<gridSize, blockSize>>> (d_destImg, interior, d_buffer_red_1, d_buffer_green_1, d_buffer_blue_1, d_blendedImg, numRowsSource, numColsSource);

   	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, numRowsSource*numColsSource * sizeof(uchar4), cudaMemcpyDeviceToHost));

	// float* to_print;
	// to_print = (float *) malloc(numRowsSource*numColsSource * sizeof(float));
	// checkCudaErrors(cudaMemcpy(to_print, d_buffer_blue_1, numRowsSource*numColsSource * sizeof(float), cudaMemcpyDeviceToHost));

	//166500

   	std::cout<<"DEBUGGING : red "<<numRowsSource*numColsSource<<std::endl;
   	for (long unsigned int i=0; i < numRowsSource*numColsSource; i++){
		if (i==86268 || i==86269 || i==86270 || i==86271 || i==86272)
			std::cout<<(int)h_blendedImg[i].x<<" "<<(int)h_blendedImg[i].y<<" "<<(int)h_blendedImg[i].z<<std::endl;
	  	// i += 1000;
   	}

   std::cout<<std::endl;
   std::cout<<std::endl;

   //Freeing all the memory
   checkCudaErrors(cudaFree(d_sourceImg));
	checkCudaErrors(cudaFree(d_destImg));
	checkCudaErrors(cudaFree(border));
	checkCudaErrors(cudaFree(interior));
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_red_source));
	checkCudaErrors(cudaFree(d_green_source));
	checkCudaErrors(cudaFree(d_blue_source));
	checkCudaErrors(cudaFree(d_buffer_red_1));
	checkCudaErrors(cudaFree(d_buffer_red_2));
	checkCudaErrors(cudaFree(d_buffer_green_1));
	checkCudaErrors(cudaFree(d_buffer_green_2));
	checkCudaErrors(cudaFree(d_buffer_blue_1));
	checkCudaErrors(cudaFree(d_buffer_blue_2));
}