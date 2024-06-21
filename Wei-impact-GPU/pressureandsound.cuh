#include <cmath>
#include <cuda_runtime.h>
#include "datatypes.cuh"


__global__ void solidsPressureAndSoundSpeedKernel(Particle* particles, int numParticles, double gamma, double sound_speed, double slope, double mi); 