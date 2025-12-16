#pragma once
#include <vector>
#include "particle.hpp"
#include "kernel.hpp"
// #include "neighbor_list.hpp"

class DensityMethod {
public:
    virtual ~DensityMethod() = default;

    virtual void compute(
        std::vector<Particle>& particles,
        std::vector<std::vector<int>>& neighbors,
        const Kernel& kernel,
        double h
    ) = 0;
};