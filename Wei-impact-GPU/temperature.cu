#include <cmath>
#include <cuda_runtime.h>
#include "datatypes.cuh"
#include "temperature.cuh"

__global__ void artificialHeatKernel(Particle* particles, InteractionPair* pairs, double* divv, int numParticles, int numPairs, double g1, double g2) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < numPairs) {
        int i = pairs[k].i;
        int j = pairs[k].j;

        double aux_divv = 0.0;
        for (int alfa = 0; alfa < 3; ++alfa) {
            double dv = particles[j].v[alfa] - particles[i].v[alfa];
            aux_divv += dv * pairs[k].dwdx[alfa];
        }

        atomicAdd(&divv[i], particles[j].m * aux_divv / particles[j].rho);
        atomicAdd(&divv[j], particles[i].m * aux_divv / particles[i].rho);
    }

    __syncthreads();

    if (k < numPairs) {
        int i = pairs[k].i;
        int j = pairs[k].j;

        double mrho = (particles[i].rho + particles[j].rho) / 2.0;
        double h_size = (particles[i].h + particles[j].h) / 2.0;

        double modr2 = 0.0;
        double rdwdx = 0.0;
        for (int alfa = 0; alfa < 3; ++alfa) {
            double dr = particles[i].r[alfa] - particles[j].r[alfa];
            modr2 += dr * dr;
            rdwdx += dr * pairs[k].dwdx[alfa];
        }

        double mui = g1 * h_size * particles[i].c + g2 * h_size * h_size * (fabs(divv[i]) - divv[i]);
        double muj = g1 * h_size * particles[j].c + g2 * h_size * h_size * (fabs(divv[j]) - divv[j]);
        double muij = (mui + muj);

        double aux = muij * rdwdx / (mrho * (modr2 + 0.01 * h_size * h_size));

        atomicAdd(&particles[i].dedt, particles[j].m * aux * (particles[i].e - particles[j].e));
        atomicAdd(&particles[j].dedt, particles[i].m * aux * (particles[j].e - particles[i].e));
    }
}
