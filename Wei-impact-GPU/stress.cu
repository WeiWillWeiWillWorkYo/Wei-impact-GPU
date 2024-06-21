#include <cmath>
#include <cuda_runtime.h>
#include "datatypes.cuh"
#include "stress.cuh"

__global__ void totalStressTensorKernel(Particle* particles, InteractionPair* pairs, int numParticles, int numPairs, double mi) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < numParticles) {
        for (int alfa = 0; alfa < 3; ++alfa) {
            for (int beta = 0; beta < 3; ++beta) {
                particles[k].epsilon[alfa][beta] = 0.0;
                particles[k].R[alfa][beta] = 0.0;
            }
        }
    }

    __syncthreads();

    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < numPairs; k += blockDim.x * gridDim.x) {
        int i = pairs[k].i;
        int j = pairs[k].j;

        double mprhoi = 0.5 * (particles[i].m / particles[i].rho);
        double mprhoj = 0.5 * (particles[j].m / particles[j].rho);

        double dv[3];
        for (int d = 0; d < 3; ++d) {
            dv[d] = particles[j].v[d] - particles[i].v[d];
        }

        for (int alfa = 0; alfa < 3; ++alfa) {
            for (int beta = 0; beta < 3; ++beta) {
                double aux_epsilon = dv[alfa] * pairs[k].dwdx[beta] + dv[beta] * pairs[k].dwdx[alfa];
                double aux_R = dv[alfa] * pairs[k].dwdx[beta] - dv[beta] * pairs[k].dwdx[alfa];

                atomicAdd(&particles[i].epsilon[alfa][beta], mprhoj * aux_epsilon);
                atomicAdd(&particles[j].epsilon[alfa][beta], mprhoi * aux_epsilon);
                atomicAdd(&particles[i].R[alfa][beta], mprhoj * aux_R);
                atomicAdd(&particles[j].R[alfa][beta], mprhoi * aux_R);
            }
        }
    }

    __syncthreads();

    if (k < numParticles) {
        double avarage = 0.0;
        for (int alfa = 0; alfa < 3; ++alfa) {
            avarage += particles[k].epsilon[alfa][alfa];
        }
        avarage /= 3.0;

        for (int alfa = 0; alfa < 3; ++alfa) {
            for (int beta = 0; beta < 3; ++beta) {
                double tau_dot = mi * particles[k].epsilon[alfa][beta];
                if (alfa == beta) {
                    tau_dot -= mi * avarage;
                }
                particles[k].tau_dot[alfa][beta] = tau_dot;

                particles[k].sigma[alfa][beta] = particles[k].tau[alfa][beta];
                if (alfa == beta) {
                    particles[k].sigma[alfa][beta] -= particles[k].p;
                }
            }
        }
    }
}

__global__ void plasticYieldModelKernel(Particle* particles, int numParticles, double J_0) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < numParticles) {
        double J = 0.0;
        for (int alfa = 0; alfa < 3; ++alfa) {
            for (int beta = 0; beta < 3; ++beta) {
                J += particles[k].tau[alfa][beta] * particles[k].tau[alfa][beta];
            }
        }
        J = sqrt(J);

        if (J > J_0) {
            for (int alfa = 0; alfa < 3; ++alfa) {
                for (int beta = 0; beta < 3; ++beta) {
                    particles[k].tau[alfa][beta] *= sqrt(J_0 / (3.0 * J * J));
                }
            }
        }
    }
}
