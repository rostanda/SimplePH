#pragma once
#include "density_method.hpp"

class SummationDensity : public DensityMethod {
public:
    void compute(
        std::vector<Particle>& p,
        std::vector<std::vector<int>>& neighbors,
        const Kernel& kernel,
        double h
    ) override
    {
        for (auto& pi : p)
            pi.rho = 0.0;

        for (size_t i = 0; i < p.size(); i++) {
            for (int j : neighbors[i]) {
                Vec2 rij = p[i].x - p[j].x;
                p[i].rho += p[j].m * kernel.W(rij, h);
            }
        }
    }
};
