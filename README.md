# parallelProgrammingSolutions
My solutions to the udacity course CS344

# Problem Set One Completed!

This problem had us do conversion of image from color to gray, which is just kind of straight-forward weighted average in mathematical terms!

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

# Problem Set Three Completed!

Another interesting computer vision appliation where we were writing a part of the algorithm known as Tone-Mapping. It normalizes the histogram of luminance to avoid problems of representing over-bright pixels on digital screens. The basic algorithm itself is very interesting.

However, for our part, we were supposed to only generate histograms and cumulative distributive function over that histogram. The second one was a clear example of Scan Primitive which we learned in our class. I wrote Hillis-Steele Scan for this, as the number of bins were quite small and it is more step-efficient.

Initially, I was using more __syncthreads(), to avoid contention in reading and writing, but then on searching for something more efficient, I stumbled across a double buffered version of Hillis-Steele scan, which works just as fine.

Generating histogram, I used the most simplest implementation using atomicAdds.

I was also trying the Method A from this paper -- https://pdfs.semanticscholar.org/2325/0770d034de0602586dc039fe1c24a6b070a8.pdf. It claimed to remove the need of atomicAdd by simulating a software mutex, which is the only bottleneck in parallel histogram algorithm. However, I think I have made some mistakes in its implementation and it didn't give me correct results. I will check that later and will move on the next lecture for now.

Another thing to note is that performing histogram on a set of data, requires finding the range, i.e. minimum and maximum in the data. It boiled down to writing a Reduce algorithm, since minimum and maximum are binary and associative operators.

I liked working on this problem as I got to learn more new and interesting things!

# All Problem Sets Completed!

Problem Set 4 was about Red-Eye Removal Algorithm of Computer Vision. But we had to implement only the sorting part where the pixels are sorted according to the scores assigned to each one of them. And I implemented LSB Parallel Radix Sort for this.

Problem Set 5 was about achieving the fastest histogram computation on our particular machine.

Problem Set 6 was about Blending Two images using Jacobi Iterative Method. The algorithm was quite simple to implement but properly writing code was necessary.