#pragma once

#include <vector>
#include <array>

#include "particle.hpp"
#include "kernel.hpp"

struct SolverState {
    // core data
    std::vector<Particle>& particles;
    std::vector<std::vector<int>>& neighbors;
    std::vector<int>& fluid_indices;
    std::vector<std::array<double,2>>& accel;

    // physics
    Kernel& kernel;
    double h;
    double mu;
    double dx0;
    double c;
    double rho0;

    // geometry
    double Lx;
    double Ly;

    // flags
    bool use_transport_velocity;
    bool use_artificial_viscosity;
    bool use_tensile_instability_correction;

    // parameters
    double alpha;
    double epsilon;

    // body force
    std::array<double,2> b_eff;
};
