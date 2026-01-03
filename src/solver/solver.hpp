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
#include "verlet_integrator.hpp"
#include "tv_verlet_integrator.hpp"
#include "vtk_writer.hpp"

class Solver
{
public:
    // constructor
    Solver(double h, double Lx, double Ly, double dx0, double Lref, double vref, KernelType kernel_type);

    void set_particles(const std::vector<Particle> &parts);
    void step(int timestep);
    void run(int steps, int vtk_freq, int log_freq);
    const std::vector<Particle> &get_particles() const;

    void set_viscosity(double mu_);
    void set_density(double rho0_, double rho_fluct_);
    void set_acceleration(const std::array<double, 2> &b_, int damp_timesteps_ = 0);
    void compute_soundspeed();
    void compute_timestep();
    void compute_timestep_AV(double mu_eff);
    void set_eos(EOSType eos_type_, double bp_fac_, double tvp_bp_fac_ = 0.0);
    void set_integrator(std::shared_ptr<Integrator> integrator_) { this->integrator = integrator_;}

    void activate_artificial_viscosity(double alpha_ = 1.0)
    {
        use_artificial_viscosity = true;
        alpha = alpha_;

        printf("artificial viscosity activated");
        
        double mu_eff = 0.0;
        if (mu==0.0)
        {
            mu_eff = rho0 *  (1.0/8.0) * alpha * h * c;
            printf("mu_eff (AV): %.6f\n", mu_eff);
        }
        if (mu>0.0)
        {
            mu_eff = rho0 * (1.0/8.0) * alpha * h * c + mu;
            printf("mu_eff (mu+AV): %.6f\n", mu_eff);
        }
        compute_timestep_AV(mu_eff);
    }

    void activate_tensile_instability_correction(double epsilon_)
    {
        use_tensile_instability_correction = true;
        epsilon = epsilon_;
            printf("tensile instability correction activated");
    }

    void activate_xsph_filter(double eta_)
    {
        use_xsph_filter = true;
        eta = eta_;
            printf("XSph filter activated (only use with velocity Verlet integrator)");
    }

    void activate_negative_pressure_truncation()
    {
        use_negative_pressure_truncation = true;
            printf("negative pressure truncation activated");
    }

    void activate_transport_velocity()
    {
        use_transport_velocity = true;
            printf("transport velocity activated");
    }


    enum class DensityMethod
    {
        Summation,
        Continuity
    };
    void set_density_method(DensityMethod density_method_) { density_method = density_method_; }

    friend void print_neighbor_counts(Solver &solver);

    int damp_timesteps;
    double mu;
    double rho0;
    double rho_fluct;

    std::array<double, 2> b = {0.0, 0.0};

private:
    double h;
    double Lx, Ly;
    double dx0;
    double Lref;
    double vref;
    double c;
    double dt;

    // effective (possibly damped) body force for the current timestep
    std::array<double, 2> compute_effective_body_force(int timestep) const;
    std::array<double, 2> b_eff = {0.0, 0.0};

    bool use_artificial_viscosity = false;
    double alpha;   // artificial viscosity alpha

    bool use_tensile_instability_correction = false;
    double epsilon; // tensile instability correction epsilon

    bool use_xsph_filter = false;
    double eta;   // xsph filter pre-factor

    bool use_negative_pressure_truncation = false;

    bool use_transport_velocity = false; // transport velocity approach

    Kernel kernel;
    Grid grid;
    EOS eos{EOSType::Tait, rho0, c, 0.0, 0.0};

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
    void compute_xsph_velocity_correction();
};
