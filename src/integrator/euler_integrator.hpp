#pragma once
#include "integrator.hpp"
#include <iostream>

class EulerIntegrator : public Integrator
{
public:
    void step1(
        std::vector<Particle> &,
        const std::vector<std::array<double, 2>> &,
        double,
        double,
        double) override
    {
        // EulerIntegrator does not have step 1
    }

    void step2(
        std::vector<Particle> &particles,
        const std::vector<std::array<double, 2>> &accel,
        double dt,
        double Lx,
        double Ly) override
    {

        for (size_t i = 0; i < particles.size(); ++i)
        {
            auto &p = particles[i];
            if (p.type == 1)
                continue;

            p.v[0] += accel[i][0] * dt;
            p.v[1] += accel[i][1] * dt;

            p.x[0] += p.v[0] * dt;
            p.x[1] += p.v[1] * dt;

            // periodic BC
            if (p.x[0] >= Lx / 2)
                p.x[0] -= Lx;
            if (p.x[0] < -Lx / 2)
                p.x[0] += Lx;
            if (p.x[1] >= Ly / 2)
                p.x[1] -= Ly;
            if (p.x[1] < -Ly / 2)
                p.x[1] += Ly;
        }
    }
};
