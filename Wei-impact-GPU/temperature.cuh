#include <cmath>
#include <cuda_runtime.h>
#include "datatypes.cuh"

__global__ void artificialHeatKernel(Particle* particles, InteractionPair* pairs, double* divv, int numParticles, int numPairs, double g1, double g2);