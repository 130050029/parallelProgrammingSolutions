# parallelProgrammingSolutions
My solutions to the udacity course CS344

# Problem Set One Completed!

I learned one interesting thing that float multiplications can differ between gpu and cpu, due to gpu trying to fusing the multiplication and add operation. And, that's why there was a minor difference when comparing with the sequential results. A thread on course forum helped me to understand the mystery. Thanks!

It happens due to CUDA, as far as I could understand. More details can be found out from here - https://docs.nvidia.com/cuda/floating-point/index.html#axzz42SnDmIrm

# Problem Set Two Completed!

This was an interesting problem where image was being blurred using a Gaussian square kernel (odd-sized for now). For us the problem was a little bit simplified by first separating the color channels, applying the kernel on each color channel and then recombining the result.

I solved this problem in four different ways!

First and foremost, getting the basic operation done, which is just straightforward multiplication and addition between kernel and every pixel's surrounding area in image. Of course, the basic idea from Problem Set One was used with a square block and corresponding mapping between threads and pixels.

Secondly, I just tried moving the filter to shared memory of every block. It gave a very tiny improvement though!

Thirdly, I tried moving the image block along with some exta elements along every edge to the shared memory space (all the elements responsible for getting output for this block of image). This gave considerable improvement!

Lastly, I combined the second and third approach and got to learn some extra syntax of shared memory. I followed this link - https://devblogs.nvidia.com/using-shared-memory-cuda-cc/.

Going through the forums I also learnt that sometimes 16x16 blockWidth can be better than 32x32, reason being that by smaller blockSizes, all the threads per block can get allocated for the task purpose. Example, if thread limit is 1536, then 16x16 fits nicely as 1536 is a multiple of 256 but 1024 is not. A good point indeed!

