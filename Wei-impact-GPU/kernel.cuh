#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <iostream>
#include <cuda_runtime.h>
#include "datatypes.cuh"


// Utility function to calculate cubic spline kernel
__device__ double cubicSplineKernel(double r, double h);

// Utility function to calculate the gradient of the cubic spline kernel
__device__ double3 cubicSplineGradient(double3 r, double h);

// Kernel function to be called in computeInteractions
__device__ void kernel(double m_dist, double3 dist, double& weight, double3& grad, double h);


#endif