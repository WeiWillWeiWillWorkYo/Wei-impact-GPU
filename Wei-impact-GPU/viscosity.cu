#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include "viscosity.cuh"
#include "differences.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "datatypes.cuh"
#include "viscosity.cuh"

void runArtificialViscosity(Particles& parts, InteractionPairs& pairs) {
    int blockSize = 256;
    int numBlocks = (pairs.quant + blockSize - 1) / blockSize;

    double alfa = 2.5;
    double beta = 2.5;

    artificialViscosityKernel << <numBlocks, blockSize >> > (parts.particle, pairs.int_pair, parts.quant, pairs.quant, alfa, beta);
    cudaDeviceSynchronize();
}
