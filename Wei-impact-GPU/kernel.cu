#include <cuda_runtime.h>
#include <math.h>
#include "kernel.cuh"
#include "constant.h"
#include "datatypes.cuh"

__device__ double cubicSplineKernel(double r, double h) {
    if (h <= 0) {
        printf("Error: h should be greater than 0.\n");
        return 0.0;
    }
    double q = r / h;
    double sigma = 1.0 / (PI * h * h * h);

    if (q <= 1.0) {
        if (q <= 0.5) {
            return sigma * (6.0 * (q * q * q - q * q) + 1.0);
        }
        else {
            return sigma * 2.0 * pow(1.0 - q, 3.0);
        }
    }
    else {
        return 0.0;
    }
}

__device__ double3 cubicSplineGradient(double3 r, double h) {
    if (h <= 0) {
        printf("Error: h should be greater than 0.\n");
        return make_double3(0.0, 0.0, 0.0);
    }
    double norm_r = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
    if (norm_r == 0.0) {
        printf("Warning: norm_r is 0.\n");
        return make_double3(0.0, 0.0, 0.0);
    }
    double q = norm_r / h;
    double sigma = 1.0 / (PI * h * h * h);
    double factor;

    if (q <= 1.0) {
        if (q <= 0.5) {
            factor = sigma * (18.0 * q - 12.0 * q * q) / h;
        }
        else {
            factor = sigma * -6.0 * (1.0 - q) * (1.0 - q) / h;
        }
        double3 grad = { r.x * factor / norm_r, r.y * factor / norm_r, r.z * factor / norm_r };
        return grad;
    }
    else {
        return make_double3(0.0, 0.0, 0.0);
    }
}

__device__ void kernel(double m_dist, double3 dist, double& weight, double3& grad, double h) {
    weight = cubicSplineKernel(m_dist, h);
    grad = cubicSplineGradient(dist, h);
    printf("Dist: %f, h: %f, Weight: %f, Grad: (%f, %f, %f)\n", m_dist, h, weight, grad.x, grad.y, grad.z);
}

