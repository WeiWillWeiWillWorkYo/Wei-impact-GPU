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

// 新增：hashFind全局函数

/*
__global__ void hashFind(Particles p, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int hash = p.hash[idx];
    int pairIndex = 0;

    printf("Particle %d, Hash: %d\n", idx, hash); // 输出当前粒子的哈希值

    // 搜索同一哈希值的粒子
    for (int j = idx + 1; j < numParticles && p.hash[j] == hash; ++j) {
        double distX = p.x[j] - p.x[idx];
        double distY = p.y[j] - p.y[idx];
        double distZ = p.z[j] - p.z[idx];
        double m_dist = sqrt(distX * distX + distY * distY + distZ * distZ);

        printf("Checking particles %d and %d, Distance: %f\n", idx, j, m_dist); // 输出距离检查信息

        double h_avg = (p.h[idx] + p.h[j]) / 2.0;
        if (m_dist < 2 * h_avg) {
            double3 dist = make_double3(distX, distY, distZ);
            double3 grad;
            double weight;
            kernel(m_dist, dist, weight, grad, h_avg);

            if (pairIndex < MAX_PAIRS) {
                p.pairs[pairIndex].i = idx;
                p.pairs[pairIndex].j = j;
                p.pairs[pairIndex].w = weight;
                p.pairs[pairIndex].grad = grad;
                pairIndex++;
            }
        }
    }
    p.numPairs[idx] = pairIndex; // 更新每个粒子的配对数
    printf("Total pairs found by particle %d: %d\n", idx, pairIndex); // 输出每个粒子找到的对数

}
*/
/*
// 05.08 updated
__global__ void hashFind(Particles p, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int hash = p.hash[idx];
    int pairIndex = 0;

    // 避免在内核中使用sqrt函数计算，直接比较距离的平方
    for (int j = idx + 1; j < numParticles && p.hash[j] == hash; ++j) {
        double distX = p.x[j] - p.x[idx];
        double distY = p.y[j] - p.y[idx];
        double distZ = p.z[j] - p.z[idx];
        double m_dist_sq = distX * distX + distY * distY + distZ * distZ;

        double h_avg = (p.h[idx] + p.h[j]) / 2.0;
        double threshold_sq = 4 * h_avg * h_avg; // 以h_avg的两倍作为距离阈值
        if (m_dist_sq < threshold_sq) {
            double3 dist = make_double3(distX, distY, distZ);
            double3 grad;
            double weight;
            kernel(sqrt(m_dist_sq), dist, weight, grad, h_avg); // 仅在必要时计算sqrt

            if (pairIndex < MAX_PAIRS) {
                p.pairs[pairIndex].i = idx;
                p.pairs[pairIndex].j = j;
                p.pairs[pairIndex].w = weight;
                p.pairs[pairIndex].grad = grad;
                pairIndex++;
            }
        }
    }
    p.numPairs[idx] = pairIndex; // 更新每个粒子的配对数
}
*/

/*
__global__ void hashFind(Particles p, int numParticles, double radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int pairIndex = 0;
    double threshold_sq = radius * radius;

    for (int j = 0; j < numParticles; j++) {
        if (idx != j) {
            double distX = p.x[j] - p.x[idx];
            double distY = p.y[j] - p.y[idx];
            double distZ = p.z[j] - p.z[idx];
            double dist_sq = distX * distX + distY * distY + distZ * distZ;
            double3 grad;
            double weight;
            if (dist_sq < threshold_sq) {
                if (pairIndex < MAX_PAIRS) {
                    p.pairs[pairIndex].i = idx;
                    p.pairs[pairIndex].j = j;
                    p.pairs[pairIndex].w = weight;
                    p.pairs[pairIndex].grad = grad;
                    pairIndex++;
                }
            }
        }
    }
    p.numPairs[idx] = pairIndex;
}
*/

/*
__global__ void findNeighbors(Particles p, Cell m_cell, double radius) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= p.count) return;

    int start_idx = m_cell.starts[m_cell.grid_hash[idx]];
    int end_idx = m_cell.ends[m_cell.grid_hash[idx]];
    int pairIndex = 0;

    for (int j = start_idx; j < end_idx; ++j) {
        int neighbor_idx = m_cell.sorted_index[j];
        if (neighbor_idx != idx && isNeighbor(p, idx, neighbor_idx, radius)) {
            double distX = p.x[neighbor_idx] - p.x[idx];
            double distY = p.y[neighbor_idx] - p.y[idx];
            double distZ = p.z[neighbor_idx] - p.z[idx];
            double dist = sqrt(distX * distX + distY * distY + distZ * distZ);
            double3 dist_vec = make_double3(distX, distY, distZ);
            double3 grad;
            double weight;
            double h_avg = (p.h[idx] + p.h[neighbor_idx]) / 2.0;

            //printf("Particle %d interacting with %d: dist = %f, h_avg = %f\n", idx, neighbor_idx, dist, h_avg);
            // 计算权重和梯度
            kernel(dist, dist_vec, weight, grad, h_avg);

            //printf("Weight: %f, Grad: (%f, %f, %f)\n", weight, grad.x, grad.y, grad.z);

            // 更新粒子属性
            if (pairIndex < MAX_PAIRS) {
                p.pairs[pairIndex].i = idx;
                p.pairs[pairIndex].j = neighbor_idx;
                p.pairs[pairIndex].w = weight;
                p.pairs[pairIndex].grad = grad;
                pairIndex++;
            }
        }
    }
    p.numPairs[idx] = pairIndex; // 更新每个粒子的配对数
}*/

__global__ void findNeighbors(Particles p, Cell m_cell, double radius) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= p.count) return;

    int start_idx = m_cell.starts[m_cell.grid_hash[idx]];
    int end_idx = m_cell.ends[m_cell.grid_hash[idx]];
    int pairIndex = 0;

    // 确保 start_idx 和 end_idx 在有效范围内
    if (start_idx == -1 || end_idx == -1 || start_idx >= end_idx) return;

    for (int j = start_idx; j < end_idx; ++j) {
        int neighbor_idx = m_cell.sorted_index[j];
        if (neighbor_idx != idx && isNeighbor(p, idx, neighbor_idx, radius)) {
            double distX = p.x[neighbor_idx] - p.x[idx];
            double distY = p.y[neighbor_idx] - p.y[idx];
            double distZ = p.z[neighbor_idx] - p.z[idx];
            double dist = sqrt(distX * distX + distY * distY + distZ * distZ);
            double3 dist_vec = make_double3(distX, distY, distZ);
            double3 grad;
            double weight;
            double h_avg = (p.h[idx] + p.h[neighbor_idx]) / 2.0;
            //printf("Particle %d interacting with %d: dist = %f, h_avg = %f\n", idx, neighbor_idx, dist, h_avg);

            // 计算权重和梯度
            kernel(dist, dist_vec, weight, grad, h_avg);

            //printf("Weight: %f, Grad: (%f, %f, %f)\n", weight, grad.x, grad.y, grad.z);
            // 更新粒子属性
            if (pairIndex < MAX_PAIRS) {
                p.pairs[pairIndex].i = idx;
                p.pairs[pairIndex].j = neighbor_idx;
                p.pairs[pairIndex].w = weight;
                p.pairs[pairIndex].grad = grad;
                pairIndex++;
            }
        }
    }
    p.numPairs[idx] = pairIndex; // 更新每个粒子的配对数

}


__global__ void computeInteractions(Particles p, double dt, double G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *p.numPairs) {
         InteractionPair pair = p.pairs[idx];
        double Fx = 0, Fy = 0, Fz = 0;

        computeGravitationalForce(Fx, Fy, Fz, G, p.mass[pair.i], p.mass[pair.j], p.x[pair.i], p.y[pair.i], p.z[pair.i], p.x[pair.j], p.y[pair.j], p.z[pair.j]);
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

__global__ void computeDensityPressure(Particles p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.count) return;

    double density = 0.0;
    for (int j = 0; j < p.count; j++) {
        if (idx != j) {
            double dx = p.x[j] - p.x[idx];
            double dy = p.y[j] - p.y[idx];
            double dz = p.z[j] - p.z[idx];
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            double h = p.h[idx];
            density += p.mass[j] * cubicSplineKernel(dist, h);
        }
    }
    p.rho[idx] = density;
    p.p[idx] = (density - p.rho_prev[idx]) * 0.5;  // 简化的压力计算
}

__global__ void computeFluidForces(Particles p, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.count) return;

    double3 force = make_double3(0.0, 0.0, 0.0);
    for (int j = 0; j < p.count; j++) {
        if (idx != j) {
            double dx = p.x[j] - p.x[idx];
            double dy = p.y[j] - p.y[idx];
            double dz = p.z[j] - p.z[idx];
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            double h = (p.h[idx] + p.h[j]) / 2.0;
            double3 grad = cubicSplineGradient(make_double3(dx, dy, dz), h);
            double weight = cubicSplineKernel(dist, h);
            force.x += weight * (p.p[idx] + p.p[j]) / (2.0 * p.rho[idx]) * grad.x;
            force.y += weight * (p.p[idx] + p.p[j]) / (2.0 * p.rho[idx]) * grad.y;
            force.z += weight * (p.p[idx] + p.p[j]) / (2.0 * p.rho[idx]) * grad.z;
        }
    }
    p.vx[idx] += force.x / p.mass[idx] * dt;
    p.vy[idx] += force.y / p.mass[idx] * dt;
    p.vz[idx] += force.z / p.mass[idx] * dt;
}