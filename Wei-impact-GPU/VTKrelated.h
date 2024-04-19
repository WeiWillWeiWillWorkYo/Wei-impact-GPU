#pragma once
#ifndef VTKRELATED_H
#define VTKRELATED_H

#include <iostream>
#include <vector>
#include "datatypes.cuh"
#include <fstream>
#include <string>
#include <iomanip> // 用于设置输出格式


void writeVTKFile(const Particles& p, int timestep) {
    std::string filename = "vtk/particles_" + std::to_string(timestep) + ".vtk";
    std::ofstream file(filename);

    if (file.is_open()) {
        file << "# vtk DataFile Version 3.0\n";
        file << "Particle data\n";
        file << "ASCII\n";
        file << "DATASET POLYDATA\n";
        file << "POINTS " << p.count << " float\n";

        for (int i = 0; i < p.count; ++i) {
            file << std::fixed << std::setprecision(6) << p.x[i] << " " << p.y[i] << " " << p.z[i] << "\n";
        }

        file << "VERTICES " << p.count << " " << 2 * p.count << "\n";
        for (int i = 0; i < p.count; ++i) {
            file << "1 " << i << "\n";
        }

        file << "POINT_DATA " << p.count << "\n";
        file << "SCALARS mass float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int i = 0; i < p.count; ++i) {
            file << p.mass[i] << "\n";
        }

        file.close();
    }
}

#endif