#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include "gravity.cuh"
#include "differences.cuh"

__device__ void computeGravitationalForce(double& Fx, double& Fy, double& Fz, double G, double mass1, double mass2, double x1, double y1, double z1, double x2, double y2, double z2) {
    double dx, dy, dz;
    computePositionDifferences(dx, dy, dz, x1, y1, z1, x2, y2, z2);
    double distSq = dx * dx + dy * dy + dz * dz + 1e-10;
    double dist = sqrt(distSq);
    double F = G * mass1 * mass2 / distSq;
    Fx += F * dx / dist;
    Fy += F * dy / dist;
    Fz += F * dz / dist;
}
