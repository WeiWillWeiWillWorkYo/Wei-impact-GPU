#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include "datatypes.cuh"
#include "integration.cuh"
#include "viscosity.cuh"
#include "gravity.cuh"
#include "differences.cuh"
#include "kernel.cuh"






__global__ void computeInteractions(Particles p, double dt, double G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *p.numPairs) {
         InteractionPair pair = p.pairs[idx];
        double Fx = 0, Fy = 0, Fz = 0;

        //computeGravitationalForce(Fx, Fy, Fz, G, p.mass[pair.i], p.mass[pair.j], p.x[pair.i], p.y[pair.i], p.z[pair.i], p.x[pair.j], p.y[pair.j], p.z[pair.j]);
        // 更新粒子i和粒子j的速度

        double ax_i = Fx / p.mass[pair.i];
        double ay_i = Fy / p.mass[pair.i];
        double az_i = Fz / p.mass[pair.i];

        // 更新粒子j的加速度（作用和反作用）
        double ax_j = -Fx / p.mass[pair.j];
        double ay_j = -Fy / p.mass[pair.j];
        double az_j = -Fz / p.mass[pair.j];

        // 根据加速度更新速度
        p.vx[pair.i] += ax_i * dt;
        p.vy[pair.i] += ay_i * dt;
        p.vz[pair.i] += az_i * dt;

        p.vx[pair.j] += ax_j * dt;
        p.vy[pair.j] += ay_j * dt;
        p.vz[pair.j] += az_j * dt;
        printf("GGGGG!!!\n");
    }
}


__global__ void handleCollisions(Particles p, double radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < p.count) {
        for (int j = 0; j < p.count; j++) {
            if (idx != j) {
                double dx = p.x[idx] - p.x[j];
                double dy = p.y[idx] - p.y[j];
                double dz = p.z[idx] - p.z[j];
                double distance = sqrt(dx * dx + dy * dy + dz * dz);
                double combinedRadius = radius; // Assuming equal radius for simplicity

                if (distance < combinedRadius) {
                    // Handle collision between particles idx and j
                    double3 vel_i = { p.vx[idx], p.vy[idx], p.vz[idx] };
                    double3 vel_j = { p.vx[j], p.vy[j], p.vz[j] };
                    handleCollision(vel_i, vel_j, { p.x[idx], p.y[idx], p.z[idx] }, { p.x[j], p.y[j], p.z[j] }, p.mass[idx], p.mass[j]);

                    // Update velocities
                    p.vx[idx] = vel_i.x;
                    p.vy[idx] = vel_i.y;
                    p.vz[idx] = vel_i.z;

                    p.vx[j] = vel_j.x;
                    p.vy[j] = vel_j.y;
                    p.vz[j] = vel_j.z;
                    //printf("GGGGG!!!\n");
                }
            }
        }
    }
}


__global__ void updatePositions(Particles p, double dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < p.count) {
            p.x[i] += p.vx[i] * dt; // 更新 x 位置
            p.y[i] += p.vy[i] * dt; // 更新 y 位置
            p.z[i] += p.vz[i] * dt; // 更新 z 位置
            printf("GGGGG!!!\n");
        }
    }

__device__ void handleCollision(double3& vel_i, double3& vel_j, double3 pos_i, double3 pos_j, double mass_i, double mass_j) {
    // 计算两粒子间的距离向量和其大小
    double3 r = { pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z };
    double dist = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
    double3 unit_r = { r.x / dist, r.y / dist, r.z / dist };

    // 计算相对速度
    double3 v_rel = { vel_j.x - vel_i.x, vel_j.y - vel_i.y, vel_j.z - vel_i.z };
    double v_rel_dot_r = v_rel.x * unit_r.x + v_rel.y * unit_r.y + v_rel.z * unit_r.z;

    // 只有当粒子相向移动时才处理碰撞
    if (v_rel_dot_r > 0) return;

    // 不完全弹性碰撞
    double e = 0.1; // 恢复系数，完全弹性碰撞为1
    double j = -(1 + e) * v_rel_dot_r / (1 / mass_i + 1 / mass_j);

    double3 impulse = { j * unit_r.x, j * unit_r.y, j * unit_r.z };

    // 根据冲量更新速度
    vel_i.x -= impulse.x / mass_i;
    vel_i.y -= impulse.y / mass_i;
    vel_i.z -= impulse.z / mass_i;

    vel_j.x += impulse.x / mass_j;
    vel_j.y += impulse.y / mass_j;
    vel_j.z += impulse.z / mass_j;
}

__global__ void computeGravitationalForce(Particles p, double G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < p.count) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (int j = 0; j < p.count; j++) {
            if (j != idx) {
                double dx = p.x[j] - p.x[idx];
                double dy = p.y[j] - p.y[idx];
                double dz = p.z[j] - p.z[idx];
                double distSq = dx * dx + dy * dy + dz * dz + 1e-10;
                double dist = sqrt(distSq);
                double force = G * p.mass[idx] * p.mass[j] / distSq;
                Fx += force * dx / dist;
                Fy += force * dy / dist;
                Fz += force * dz / dist;
            }
        }

        p.vx[idx] += Fx / p.mass[idx] * G;
        p.vy[idx] += Fy / p.mass[idx] * G;
        p.vz[idx] += Fz / p.mass[idx] * G;
    }
}

__global__ void applyBoundaryConditions(Particles p, double boxSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < p.count) {
        if (p.x[idx] < 0.0) {
            p.x[idx] = 0.0;
            p.vx[idx] *= -1;
        }
        if (p.x[idx] > boxSize) {
            p.x[idx] = boxSize;
            p.vx[idx] *= -1;
        }
        if (p.y[idx] < 0.0) {
            p.y[idx] = 0.0;
            p.vy[idx] *= -1;
        }
        if (p.y[idx] > boxSize) {
            p.y[idx] = boxSize;
            p.vy[idx] *= -1;
        }
        if (p.z[idx] < 0.0) {
            p.z[idx] = 0.0;
            p.vz[idx] *= -1;
        }
        if (p.z[idx] > boxSize) {
            p.z[idx] = boxSize;
            p.vz[idx] *= -1;
        }
    }
}
