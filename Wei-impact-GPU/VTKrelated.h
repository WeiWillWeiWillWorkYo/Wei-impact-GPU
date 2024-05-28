#pragma once
#ifndef VTKRELATED_H
#define VTKRELATED_H

#include <iostream>
#include <vector>
#include "datatypes.cuh"
#include <fstream>
#include <string>
#include <iomanip> // 用于设置输出格式
#include <set>


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

void writeVTKWithDifferentColors(const Particles& p, int selected_index, const std::vector<int>& neighbors, int timestep) {
    std::string filename = "vtk/all_particles_" + std::to_string(timestep) + ".vtk";
    std::ofstream file(filename);

    if (file.is_open()) {
        int numPoints = p.count;  // Total number of particles

        file << "# vtk DataFile Version 3.0\n";
        file << "All particles data with selected and neighbors highlighted\n";
        file << "ASCII\n";
        file << "DATASET POLYDATA\n";
        file << "POINTS " << numPoints << " float\n";

        // Output all particles' positions
        for (int i = 0; i < numPoints; ++i) {
            file << std::fixed << std::setprecision(6) << p.x[i] << " " << p.y[i] << " " << p.z[i] << "\n";
        }

        file << "VERTICES " << numPoints << " " << 2 * numPoints << "\n";
        for (int i = 0; i < numPoints; ++i) {
            file << "1 " << i << "\n";
        }

        file << "POINT_DATA " << numPoints << "\n";
        file << "SCALARS color float 1\n";
        file << "LOOKUP_TABLE default\n";

        for (int i = 0; i < numPoints; ++i) {
            if (i == selected_index) {
                file << "1.0\n";  // Unique highlight color for the selected particle
            }
            else if (std::find(neighbors.begin(), neighbors.end(), i) != neighbors.end()) {
                file << "0.5\n";  // Different highlight color for neighbors
            }
            else {
                file << "0.0\n";  // Default color for other particles
            }
        }

        file.close();
    }
    else {
        std::cerr << "Failed to open file for writing VTK data.\n";
    }
}

#endif