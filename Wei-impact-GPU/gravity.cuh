#pragma once
#ifndef GRAVITY_CUH
#define GRAVITY_CUH

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>


__device__ void computeGravitationalForce(double& Fx, double& Fy, double& Fz, double G, double mass1, double mass2, double x1, double y1, double z1, double x2, double y2, double z2);

#endif