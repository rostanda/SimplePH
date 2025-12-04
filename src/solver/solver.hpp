#pragma once
#include <vector>
#include <memory>
#include <chrono>
#include "particle.hpp"
#include "grid.hpp"
#include "kernel.hpp"
#include "eos.hpp"
#include "integrator.hpp"
#include "euler_integrator.hpp"
#include "velocityverlet_integrator.hpp"
#include "vtk_writer.hpp"

class Solver
{
public:
    // constructor
    Solver(double h, double Lx, double Ly, double dx0, double Lref, double vref, KernelType kernel_type);

    void set_particles(const std::vector<Particle> &parts);
    void step();
    void run(int steps, int vtk_freq, int log_freq);
    const std::vector<Particle> &get_particles() const;

    void set_viscosity(double mu_);
    void set_density(double rho0_, double rho_fluct_);
    void set_acceleration(const std::array<double, 2> &b_);
    void compute_soundspeed();
    void compute_timestep();
    void set_eos(EOSType eos_type_, double bp_);
    void set_integrator(std::shared_ptr<Integrator> integrator_) { this->integrator = integrator_; }

    enum class DensityMethod
    {
        Summation,
        Continuity
    };
    void set_density_method(DensityMethod density_method_) { density_method = density_method_; }

    friend void print_neighbor_counts(Solver &solver);

    std::array<double, 2> b = {0.0, 0.0};
    double mu;
    double rho0;
    double rho_fluct;

private:
    double h;
    double Lx, Ly;
    double dx0;
    double Lref;
    double vref;
    double c;
    double dt;

    Kernel kernel;
    Grid grid;
    EOS eos{EOSType::Tait, rho0, c, 0.0};

    std::shared_ptr<Integrator> integrator;

    std::vector<Particle> particles;          // particle vector
    std::vector<std::vector<int>> neighbors;  // neighborlist per particle
    std::vector<std::array<double, 2>> accel; // acceleration

    // particle indices by type
    std::vector<int> fluid_indices;
    std::vector<int> boundary_indices;
    // std::vector<int> solid_indices; // for future use
    void build_index_lists();


    DensityMethod density_method = DensityMethod::Summation;

    double total_elapsed_time = 0.0; // total simulation time in seconds

    void update_neighbors();
    void compute_density_summation();
    void compute_density_continuity();
    void compute_pressure();
    void compute_boundaryconditions();
    void compute_forces();
};
