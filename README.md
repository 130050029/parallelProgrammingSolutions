# parallelProgrammingSolutions
My solutions to the udacity course CS344

# Problem Set One done for now

I learned one interesting thing that float multiplications can differ between gpu and cpu, due to gpu trying to fusing the multiplication and add operation. And, that's why there was a minor difference when comparing with the sequential results. A thread on course forum helped me to understand the mystery. Thanks!

It happens due to CUDA, as far as I could understand. More details can be found out from here - https://docs.nvidia.com/cuda/floating-point/index.html#axzz42SnDmIrm
