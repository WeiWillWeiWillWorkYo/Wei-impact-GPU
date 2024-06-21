#include <iostream>
#include <cuda_runtime.h>
#include "datatypes.cuh"
#include "density.cuh"

__global__ void normSumDensityKernel(Particle* particles, InteractionPair* pairs, double* wi, int numParticles, int numPairs) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < numParticles) {
        wi[k] = weight(0.0) * particles[k].m / particles[k].rho;
    }
    __syncthreads();

    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < numPairs; k += blockDim.x * gridDim.x) {
        int i = pairs[k].i;
        int j = pairs[k].j;
        atomicAdd(&wi[i], pairs[k].w * particles[i].m / particles[i].rho);
        atomicAdd(&wi[j], pairs[k].w * particles[j].m / particles[j].rho);
    }

    __syncthreads();

    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < numParticles; k += blockDim.x * gridDim.x) {
        particles[k].rho /= wi[k];
    }
}

__global__ void sumDensityKernel(Particle* particles, InteractionPair* pairs, int numParticles, int numPairs) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < numParticles) {
        particles[k].rho = weight(0.0) * particles[k].m;
    }
    __syncthreads();

    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < numPairs; k += blockDim.x * gridDim.x) {
        int i = pairs[k].i;
        int j = pairs[k].j;
        atomicAdd(&particles[i].rho, particles[i].m * pairs[k].w);
        atomicAdd(&particles[j].rho, particles[j].m * pairs[k].w);
    }
}

__global__ void conDensityKernel(Particle* particles, InteractionPair* pairs, int numParticles, int numPairs) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < numParticles) {
        particles[k].drhodt = 0.0;
    }
    __syncthreads();

    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < numPairs; k += blockDim.x * gridDim.x) {
        int i = pairs[k].i;
        int j = pairs[k].j;
        double vcc = 0.0;
        for (int beta = 0; beta < 3; ++beta) {
            double dv = particles[i].v[beta] - particles[j].v[beta];
            vcc += dv * pairs[k].dwdx[beta];
        }
        atomicAdd(&particles[i].drhodt, particles[j].m * vcc);
        atomicAdd(&particles[j].drhodt, particles[i].m * vcc);
    }
}

__global__ void copy2InitDensityKernel(Particle* particles, int numParticles) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < numParticles) {
        particles[k].rho_0 = particles[k].rho;
    }
}