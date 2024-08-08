// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "utils.h"
#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>


#define gpuErrchk() { gpuAssert(__FILE__, __LINE__); }
inline void gpuAssert(const char* file, int line, bool abort = true)
{
    cudaDeviceSynchronize();
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}





__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  // TODO
  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
  
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
  int outputRow = threadIdx.y + blockIdx.y * blockDim.y;
  int outputCol = threadIdx.x + blockIdx.x * blockDim.x;

  if (outputRow >= numRows || outputCol >=numCols){
    return;
  }

  unsigned int image1DIndexOutput = numCols * outputRow + outputCol;
  
  float outputVal = 0.0f;

  for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
    for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
      
      int image_r = outputRow + filter_r;
      image_r = image_r > 0 ? image_r :0;
      image_r = image_r < numRows ? image_r : numRows -1;     

      int image_c = outputCol + filter_c;
      image_c = image_c > 0 ? image_c :0;
      image_c = image_c < numCols ? image_c : numCols -1;

      int image1DIdx = numCols * image_r + image_c;
      int filter1DIdx = (filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2;

      outputVal += inputChannel[image1DIdx] * filter[filter1DIdx];
    }
  }

  outputChannel[image1DIndexOutput] = outputVal;

}

__global__
void gaussian_blur_sh_filter(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  // TODO
  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
  
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
  int outputRow = threadIdx.y + blockIdx.y * blockDim.y;
  int outputCol = threadIdx.x + blockIdx.x * blockDim.x;

  if (outputRow >= numRows || outputCol >=numCols){
    return;
  }

  extern __shared__ float sh_filter[];

  int filter_r = threadIdx.y;
  int filter_c = threadIdx.x;

  if (filter_r < filterWidth && filter_c < filterWidth){
    //for (int filter_r = 0; filter_r < filterWidth; ++filter_r) {
    //  for (int filter_c = 0; filter_c < filterWidth; ++filter_c) {
        int filterIdx = filter_r * filterWidth + filter_c;
        sh_filter[filterIdx] = filter[filterIdx];
    //  }
    //}
  }

  __syncthreads();

  unsigned int image1DIndexOutput = numCols * outputRow + outputCol;
  
  float outputVal = 0.0f;

  for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
    for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
      
      int image_r = outputRow + filter_r;
      image_r = image_r > 0 ? image_r :0;
      image_r = image_r < numRows ? image_r : numRows -1;     

      int image_c = outputCol + filter_c;
      image_c = image_c > 0 ? image_c :0;
      image_c = image_c < numCols ? image_c : numCols -1;

      int image1DIdx = numCols * image_r + image_c;
      int filter1DIdx = (filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2;

      outputVal += inputChannel[image1DIdx] * sh_filter[filter1DIdx];
    }
  }

  outputChannel[image1DIndexOutput] = outputVal;

}

__device__
int clamp(int a, int lo, int hi){
   int ret;
   ret = lo <= a ? a : lo;
   ret = ret < hi ? ret : hi -1;
   return ret;
}

__global__
void gaussian_blur_sh_block_image(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  // TODO
  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
  
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
  int outputRow = threadIdx.y + blockIdx.y * blockDim.y;
  int outputCol = threadIdx.x + blockIdx.x * blockDim.x;

  unsigned int image1DIndexOutput = outputRow * numCols + outputCol;

  extern __shared__ unsigned char sh_img_block[];

  //shared memory is a block of square of size = blocksize + filterWidth -1;
  int sh_mem_block_size = blockDim.x + filterWidth-1;
  int halfWidth = filterWidth/2;

  int ori_image_row_pos = outputRow - halfWidth;
  int ori_image_col_pos = outputCol - halfWidth;

  int sh_img_row_pos = threadIdx.y;
  int sh_img_col_pos = threadIdx.x;

  ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
  ori_image_col_pos = clamp(ori_image_col_pos, 0, numCols);

  // current pixel mapped to shared_memory location!
  sh_img_block[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos] = inputChannel[ori_image_row_pos * numCols + ori_image_col_pos];

  int boundary_thresh_row = blockDim.y - filterWidth+1;
  int boundary_thresh_col = blockDim.x - filterWidth+1;

  if (threadIdx.y >= boundary_thresh_row){
    // filling in the lower bottom rectangle of size (filterWidth -1 ) * (blockDim.x) of shared memory except for bottom-right corner
    ori_image_row_pos = outputRow + halfWidth;
    ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
    sh_img_block[(sh_img_row_pos + filterWidth -1)* sh_mem_block_size + sh_img_col_pos] = inputChannel[ori_image_row_pos * numCols + ori_image_col_pos];
  }

  if (threadIdx.x >= boundary_thresh_col){
    // filling in the right left rectangle of size (blockDim.y) * (filterWidth - 1) of shared memory except for bottom-right corner
    ori_image_row_pos = outputRow - halfWidth;
    ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
    ori_image_col_pos = outputCol + halfWidth;
    ori_image_col_pos = clamp(ori_image_col_pos, 0, numCols);
    sh_img_block[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos + filterWidth-1] = inputChannel[ori_image_row_pos * numCols + ori_image_col_pos];
  }
  
  if (threadIdx.x < (filterWidth - 1) && threadIdx.y < (filterWidth-1)) {
    // filling the bottom-right corner of zie (filterWidth - 1) ^ (2).
    ori_image_row_pos = outputRow - halfWidth + blockDim.y;
    ori_image_col_pos = outputCol - halfWidth + blockDim.x;

    ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
    ori_image_col_pos = clamp(ori_image_col_pos, 0, numCols);

    sh_img_block[(sh_img_row_pos + blockDim.y)*sh_mem_block_size + sh_img_col_pos + blockDim.x] = inputChannel[ori_image_row_pos * numCols + ori_image_col_pos];
  }

  __syncthreads();

  if (outputRow >= numRows || outputCol >=numCols){
    return;
  }
  
  float outputVal = 0.0f;

  for (int filter_r = 0; filter_r < filterWidth; filter_r++) {
    for (int filter_c = 0; filter_c < filterWidth; filter_c++) {
      int image_r, image_c;
      image_r = threadIdx.y + filter_r;
      image_c = threadIdx.x + filter_c;
      float imageVal = static_cast<float>(sh_img_block[image_r * sh_mem_block_size + image_c]);
      //float imageVal = sh_img_block[image_r * sh_mem_block_size + image_c]; 
      outputVal += filter[filter_r * filterWidth + filter_c] * imageVal;
      //outputVal += sh_img_block[image_r * sh_mem_block_size + image_c] * filter[filter_r * filterWidth + filter_c];
    }
  }

  outputChannel[image1DIndexOutput] = outputVal;

}


// Function with both filter and image block moved to shared memory!!
__global__
void gaussian_blur_sh_filter_image(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  // TODO
  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
  
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
  int outputRow = threadIdx.y + blockIdx.y * blockDim.y;
  int outputCol = threadIdx.x + blockIdx.x * blockDim.x;

  unsigned int image1DIndexOutput = outputRow * numCols + outputCol;

  int sh_mem_block_size = blockDim.x + filterWidth-1;

  extern __shared__ int sh_mem[];
  int *sh_pointer = sh_mem;

  float *sh_filter = (float*)&sh_pointer[0]; // first portion of memory starts from position to 0 and given to filter
  unsigned char *sh_img_block = (unsigned char*)&sh_filter[filterWidth*filterWidth]; // fW * fW * 4 bytes positions taken by filter

  //shared memory is a block of square of size = blocksize + filterWidth -1;
  int halfWidth = filterWidth/2;

  int ori_image_row_pos = outputRow - halfWidth;
  int ori_image_col_pos = outputCol - halfWidth;

  int sh_img_row_pos = threadIdx.y;
  int sh_img_col_pos = threadIdx.x;

  ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
  ori_image_col_pos = clamp(ori_image_col_pos, 0, numCols);

  // current pixel mapped to shared_memory location!
  sh_img_block[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos] = inputChannel[ori_image_row_pos * numCols + ori_image_col_pos];

  int boundary_thresh_row = blockDim.y - filterWidth+1;
  int boundary_thresh_col = blockDim.x - filterWidth+1;

  if (threadIdx.y >= boundary_thresh_row){
    // filling in the lower bottom rectangle of size (filterWidth -1 ) * (blockDim.x) of shared memory except for bottom-right corner
    ori_image_row_pos = outputRow + halfWidth;
    ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
    sh_img_block[(sh_img_row_pos + filterWidth -1)* sh_mem_block_size + sh_img_col_pos] = inputChannel[ori_image_row_pos * numCols + ori_image_col_pos];
  }

  if (threadIdx.x >= boundary_thresh_col){
    // filling in the right left rectangle of size (blockDim.y) * (filterWidth - 1) of shared memory except for bottom-right corner
    ori_image_row_pos = outputRow - halfWidth;
    ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
    ori_image_col_pos = outputCol + halfWidth;
    ori_image_col_pos = clamp(ori_image_col_pos, 0, numCols);
    sh_img_block[sh_img_row_pos * sh_mem_block_size + sh_img_col_pos + filterWidth-1] = inputChannel[ori_image_row_pos * numCols + ori_image_col_pos];
  }
  
  if (threadIdx.x < (filterWidth - 1) && threadIdx.y < (filterWidth-1)) {
    // filling the bottom-right corner of zie (filterWidth - 1) ^ (2).
    ori_image_row_pos = outputRow - halfWidth + blockDim.y;
    ori_image_col_pos = outputCol - halfWidth + blockDim.x;

    ori_image_row_pos = clamp(ori_image_row_pos, 0, numRows);
    ori_image_col_pos = clamp(ori_image_col_pos, 0, numCols);

    sh_img_block[(sh_img_row_pos + blockDim.y)*sh_mem_block_size + sh_img_col_pos + blockDim.x] = inputChannel[ori_image_row_pos * numCols + ori_image_col_pos];
  }

  if (threadIdx.x < filterWidth && threadIdx.y < filterWidth) {
    int filter1Dpos = threadIdx.y * filterWidth + threadIdx.x;
    sh_filter[filter1Dpos] = filter[filter1Dpos];
  }

  __syncthreads();

  if (outputRow >= numRows || outputCol >=numCols){
    return;
  }
  
  float outputVal = 0.0f;

  for (int filter_r = 0; filter_r < filterWidth; filter_r++) {
    for (int filter_c = 0; filter_c < filterWidth; filter_c++) {
      int image_r, image_c;
      image_r = threadIdx.y + filter_r;
      image_c = threadIdx.x + filter_c;
      float imageVal = static_cast<float>(sh_img_block[image_r * sh_mem_block_size + image_c]);
      //float imageVal = sh_img_block[image_r * sh_mem_block_size + image_c]; 
      outputVal += sh_filter[filter_r * filterWidth + filter_c] * imageVal;
      //outputVal += sh_img_block[image_r * sh_mem_block_size + image_c] * filter[filter_r * filterWidth + filter_c];
    }
  }

  outputChannel[image1DIndexOutput] = outputVal;

}


//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
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
  uchar4 rgba = inputImageRGBA[image1DIndex];

  redChannel[image1DIndex] = rgba.x;
  greenChannel[image1DIndex] = rgba.y;
  blueChannel[image1DIndex] = rgba.z;

}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc

  unsigned int FILTER_BYTES_SIZE = sizeof(float) * filterWidth * filterWidth;

  checkCudaErrors(cudaMalloc(&d_filter, FILTER_BYTES_SIZE));

  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!

  checkCudaErrors(cudaMemcpy(d_filter, h_filter, FILTER_BYTES_SIZE, cudaMemcpyHostToDevice));

}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
//void your_gaussian_blur(uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  int blockWidth = 16;
  int blocksX, blocksY;
 
  if (numCols%blockWidth==0){
     blocksX = numCols/blockWidth;
  } else {
     blocksX = (numCols/blockWidth) + 1;
  }

  if (numRows%blockWidth==0){
     blocksY = numRows/blockWidth;
  } else {
     blocksY = (numRows/blockWidth) + 1;
  }

  //TODO: Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(blockWidth, blockWidth,1);

  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize(blocksX, blocksY,1);

  //TODO: Launch a kernel for separating the RGBA image into different color channels

  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());
//  gpuErrchk();


  //TODO: Call your convolution kernel here 3 times, once for each color channel.

  int SH_MEM_FILTER_BYTES_SIZE = sizeof(float) * filterWidth * filterWidth;
  int SH_MEM_IMAGE_BYTES_SIZE = sizeof(unsigned char) * (blockWidth + filterWidth - 1) * (blockWidth + filterWidth - 1);

  //gaussian_blur_sh_filter<<<gridSize, blockSize, SH_MEM_FILTER_BYTES_SIZE>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  //gaussian_blur_sh_filter<<<gridSize, blockSize, SH_MEM_FILTER_BYTES_SIZE>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  //gaussian_blur_sh_filter<<<gridSize, blockSize, SH_MEM_FILTER_BYTES_SIZE>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

  // Call when no shared memory is used
  //gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  //gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  //gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);


  // Call when only Image Block is shared
  //gaussian_blur_sh_block_image<<<gridSize, blockSize, SH_MEM_IMAGE_BYTES_SIZE>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  //gaussian_blur_sh_block_image<<<gridSize, blockSize, SH_MEM_IMAGE_BYTES_SIZE>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  //gaussian_blur_sh_block_image<<<gridSize, blockSize, SH_MEM_IMAGE_BYTES_SIZE>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

  // Call when Filter and Image Block both are shared.
  gaussian_blur_sh_filter_image<<<gridSize, blockSize, SH_MEM_IMAGE_BYTES_SIZE + SH_MEM_FILTER_BYTES_SIZE>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur_sh_filter_image<<<gridSize, blockSize, SH_MEM_IMAGE_BYTES_SIZE + SH_MEM_FILTER_BYTES_SIZE>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur_sh_filter_image<<<gridSize, blockSize, SH_MEM_IMAGE_BYTES_SIZE + SH_MEM_FILTER_BYTES_SIZE>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
