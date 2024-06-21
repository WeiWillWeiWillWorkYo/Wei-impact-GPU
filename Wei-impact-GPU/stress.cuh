#include <cmath>
#include <cuda_runtime.h>
#include "datatypes.cuh"
#include "datatypes.cuh"

__global__ void totalStressTensorKernel(Particle* particles, InteractionPair* pairs, int numParticles, int numPairs, double mi);
__global__ void plasticYieldModelKernel(Particle* particles, int numParticles, double J_0);