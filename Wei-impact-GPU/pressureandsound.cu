#include <cmath>
#include <cuda_runtime.h>
#include "datatypes.cuh"
#include "pressureandsound.cuh"

__global__ void solidsPressureAndSoundSpeedKernel(Particle* particles, int numParticles, double gamma, double sound_speed, double slope, double mi) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < numParticles) {
        double a_0 = particles[k].rho_0 * sound_speed * sound_speed;
        double b_0 = a_0 * (1.0 + 2.0 * (slope - 1.0));
        double c_0 = a_0 * (2.0 * (slope - 1.0) + 3.0 * (slope - 1.0) * (slope - 1.0));

        double eta = (particles[k].rho / particles[k].rho_0) - 1.0;

        double p_H;
        if (eta > 0.0) {
            p_H = a_0 * eta + b_0 * eta * eta + c_0 * eta * eta * eta;
        }
        else {
            p_H = a_0 * eta;
        }

        particles[k].p = (1.0 - gamma * eta / 2.0) * p_H + gamma * particles[k].rho * particles[k].e;
        particles[k].c = sqrt(4.0 * mi / (3.0 * particles[k].rho_0));
    }
}
