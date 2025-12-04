#pragma once
#include "integrator.hpp"

class VelocityVerletIntegrator : public Integrator
{
public:
    void step1(
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

            p.v[0] += accel[i][0] * dt * 0.5;
            p.v[1] += accel[i][1] * dt * 0.5;

            p.x[0] += p.v[0] * dt;
            p.x[1] += p.v[1] * dt;

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

            p.v[0] += accel[i][0] * dt * 0.5;
            p.v[1] += accel[i][1] * dt * 0.5;
        }
    }
};
