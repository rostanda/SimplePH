#include <cstdio>
#include <algorithm>
#include <cmath>
#include <limits>
#include <deque>
#include "solver.hpp"
#include "kernel.hpp"
#include "utils.hpp"
#include "integrator.hpp"
#include "terminal.hpp"

// include OpenMP for parallelization
#include <omp.h>

// constructor
Solver::Solver(double h_, double Lx_, double Ly_, double dx0_, double Lref_, double vref_, KernelType kernel_type_)
    : h(h_), Lx(Lx_), Ly(Ly_), dx0(dx0_), Lref(Lref_), vref(vref_), kernel(kernel_type_, h_),
      grid(kernel.get_rcut(h_), Lx_, Ly_)
{
#pragma omp parallel
    {
#pragma omp single
        {
            printf("[OpenMP] Threads in this parallel region: %d\n", omp_get_num_threads());
        }
    }
}

// set particles
void Solver::set_particles(const std::vector<Particle> &parts)
{
    particles = parts;
    neighbors.resize(particles.size());
    accel.resize(particles.size(), {0.0, 0.0});
    build_index_lists();

    // initialize BC particles
    for (auto &p : particles)
    {
        if (p.type == 1)
            p.vf = {0.0, 0.0};
        else
            p.vf = std::nullopt;
    }

    for (auto &p : particles)
    {
        if (density_method == DensityMethod::Summation)
            p.drho_dt = std::nullopt;
        else if (density_method == DensityMethod::Continuity)
            p.drho_dt = 0.0;

        if (use_xsph_filter)
            p.v_xsph = {0.0, 0.0};
        else
            p.v_xsph = std::nullopt;

        if (use_transport_velocity)
        {
            p.tv = {0.0, 0.0};
            p.bpc = {0.0, 0.0};
        }
        else
        {
            p.tv = std::nullopt;
            p.bpc = std::nullopt;
        }
    }
}

// build index lists for particle types
void Solver::build_index_lists()
{
    fluid_indices.clear();
    boundary_indices.clear();
    // solid_indices.clear(); // for future use

    for (int i = 0; i < particles.size(); ++i)
    {
        if (particles[i].type == 0)
            fluid_indices.push_back(i);
        else if (particles[i].type == 1)
            boundary_indices.push_back(i);
        // else if (particles[i].type == 2) // for future use
        //     solid_indices.push_back(i);
    }
}

// set viscosity
void Solver::set_viscosity(double mu_) { mu = mu_; }

// set density
void Solver::set_density(double rho0_, double rho_fluct_)
{
    rho0 = rho0_;
    rho_fluct = rho_fluct_;
}

// set body force
void Solver::set_acceleration(const std::array<double, 2> &b_, int damp_timesteps_)
{
    b = b_;
    damp_timesteps = damp_timesteps_;
}

// compute soundspeed
void Solver::compute_soundspeed()
{
    double c_cfl = (vref * vref) / rho_fluct;
    double b_mag = std::sqrt(b[0] * b[0] + b[1] * b[1]);
    double c_gw = (b_mag * Lref) / rho_fluct;
    double c_fc = (mu * vref) / (rho_fluct * rho0 * Lref);
    c = std::sqrt(std::max({c_cfl, c_gw, c_fc}));
    printf("soundspeed: c=%.3f\n", c);
}

// compute timestep
void Solver::compute_timestep()
{
    std::vector<double> candidates;
    candidates.push_back(0.25 * h / c); // CFL
    double b_mag = std::sqrt(b[0] * b[0] + b[1] * b[1]);
    if (b_mag > 0.0)
        candidates.push_back(std::sqrt(h / (16.0 * b_mag))); // gravity-wave
    candidates.push_back((h * h * rho0) / (8.0 * mu));       // Fourier
    dt = *std::min_element(candidates.begin(), candidates.end());
    printf("timestep: dt=%.9f\n", dt);
}

// compute timestep
void Solver::compute_timestep_AV(double mu_eff)
{
    std::vector<double> candidates;
    candidates.push_back(0.25 * h / c); // CFL
    double b_mag = std::sqrt(b[0] * b[0] + b[1] * b[1]);
    if (b_mag > 0.0)
        candidates.push_back(std::sqrt(h / (16.0 * b_mag))); // gravity-wave
    candidates.push_back((h * h * rho0) / (8.0 * mu_eff));   // Fourier
    dt = *std::min_element(candidates.begin(), candidates.end());
    printf("timestep: dt=%.9f\n", dt);
}

// set EOS
void Solver::set_eos(EOSType eos_type_, double bp_fac_, double tvp_bp_fac_) { eos = EOS(eos_type_, rho0, c, bp_fac_, tvp_bp_fac_); }

// get particles
const std::vector<Particle> &Solver::get_particles() const { return particles; }

// solve one timestep
void Solver::step(int timestep)
{

    // compute effective (possibly damped) body force for this timestep
    this->b_eff = compute_effective_body_force(timestep);
    integrator->step1(particles, fluid_indices, accel, dt, Lx, Ly);
    update_neighbors();
    if (density_method == DensityMethod::Summation)
        compute_density_summation();
    compute_pressure();
    compute_boundaryconditions();
    if (use_xsph_filter)
        compute_xsph_velocity_correction();
    compute_forces();
    if (density_method == DensityMethod::Continuity)
        compute_density_continuity();
    integrator->step2(particles, fluid_indices, accel, dt, Lx, Ly);
}

// run simulation
void Solver::run(int steps, int vtk_freq, int log_freq)
{
    std::cout << "\n===== Start simulation =====" << std::endl;

    bool first_log = true;
    constexpr int STATUS_LINES = 2;

    if (vtk_freq <= 0)
        vtk_freq = 1;
    if (log_freq < 0)
        log_freq = 0;

    int vtk_counter = 0, log_counter = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    const int eta_window = 50;
    std::deque<double> step_times;

    for (int i = 0; i < steps; ++i)
    {
        auto step_start = std::chrono::high_resolution_clock::now();

        if (vtk_counter == 0)
            VTKWriter::write(particles, i);
        vtk_counter = (vtk_counter + 1) % vtk_freq;
        if (log_freq > 0)
            log_counter = (log_counter + 1) % log_freq;
        step(i);

        auto step_end = std::chrono::high_resolution_clock::now();
        double step_time = std::chrono::duration<double>(step_end - step_start).count();

        step_times.push_back(step_time);
        if (step_times.size() > eta_window)
            step_times.pop_front();

        double avg_step_time = 0.0;
        for (double t : step_times)
            avg_step_time += t;
        avg_step_time /= step_times.size();

        int remaining_steps = steps - (i + 1);
        double eta_sec = remaining_steps * avg_step_time;

        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - start_time).count();

        if (log_freq > 0 && log_counter == 0)
        {
            if (!first_log)
                terminal::clear_lines(STATUS_LINES);
            first_log = false;

            double progress = 100.0 * (i + 1) / steps;

            std::cout
                << "Step " << (i + 1) << " / " << steps
                << " (" << std::fixed << std::setprecision(1) << progress << "%)\n"
                << "iter: " << std::setprecision(4) << step_time << " s"
                << " | avg: " << avg_step_time << " s"
                << " | ETA: " << format_time(eta_sec) << "\n";

            terminal::flush();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    total_elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "\n=== Simulation completed ===" << std::endl;
    std::cout << "Total time: " << total_elapsed_time << "s" << std::endl;
    std::cout << "Total steps: " << steps << std::endl;
    std::cout << "Average time per iter: " << (total_elapsed_time / steps) << "s" << std::endl;
}

// update neighbor list
void Solver::update_neighbors()
{
    neighbors.clear();
    neighbors.resize(particles.size());
    grid.build(particles);

#pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i)
    {
        grid.find_neighbors(static_cast<int>(i), particles,
                            [&](int pid, int jid, double dx, double dy, double r)
                            { neighbors[pid].push_back(jid); });
    }
}

std::array<double, 2> Solver::compute_effective_body_force(int timestep) const
{
    std::array<double, 2> b_eff_local = b; // Basis-Body-Force
    if (damp_timesteps > 0 && timestep < damp_timesteps)
    {
        double t = static_cast<double>(timestep);
        double m_damptime = static_cast<double>(damp_timesteps);
        double scalar = 0.5 * (std::sin(M_PI * (-0.5 + t / m_damptime)) + 1.0);
        b_eff_local[0] *= scalar;
        b_eff_local[1] *= scalar;
    }
    return b_eff_local;
}

// compute density summation (rho_i = sum_j m_j W(r_ij))
void Solver::compute_density_summation()
{
// only loop over fluid particles
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < (int)fluid_indices.size(); ++idx)
    {
        int i = fluid_indices[idx];
        Particle &pi = particles[i];
        double rho = 0.0;

        for (int j : neighbors[i])
        {
            Particle &pj = particles[j];
            double dx = pi.x[0] - pj.x[0];
            double dy = pi.x[1] - pj.x[1];
            double r = min_image_dist(dx, dy, Lx, Ly);
            rho += pj.m * kernel.getW(r, h);
        }
        // self-contribution
        rho += pi.m * kernel.getW(0.0, h);
        pi.rho = rho;
    }
}

// compute density using continuity equation (d rho / dt)
void Solver::compute_density_continuity()
{
#pragma omp parallel for schedule(static)
    for (auto &p : particles)
        p.drho_dt = 0.0;

// only loop over fluid particles
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < (int)fluid_indices.size(); ++idx)
    {
        int i = fluid_indices[idx];
        Particle &pi = particles[i];

        double drho_dt_i = 0.0;
        const double vx_i = pi.v[0];
        const double vy_i = pi.v[1];
        const double rho_i = pi.rho;

        for (int j : neighbors[i])
        {
            Particle &pj = particles[j];

            double dx = pi.x[0] - pj.x[0];
            double dy = pi.x[1] - pj.x[1];
            double r = min_image_dist(dx, dy, Lx, Ly);
            double dW = kernel.getdW(r, h) / r;

            const double vx_j = (pj.type == 1 && pj.vf.has_value()) ? (*pj.vf)[0] : pj.v[0];
            const double vy_j = (pj.type == 1 && pj.vf.has_value()) ? (*pj.vf)[1] : pj.v[1];

            const double dvx = vx_i - vx_j;
            const double dvy = vy_i - vy_j;
            drho_dt_i += (pj.m / pj.rho) * (dvx * dW * dx + dvy * dW * dy);
        }
        drho_dt_i *= rho_i;
        pi.drho_dt = drho_dt_i;
        pi.rho = rho_i + drho_dt_i * dt;
    }
}

// compute pressure
void Solver::compute_pressure()
{

// only loop over fluid particles
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < (int)fluid_indices.size(); ++idx)
    {
        int i = fluid_indices[idx];
        Particle &pi = particles[i];
        // pi.p = eos.pressure_from_density(pi.rho);
        double p = eos.pressure_from_density(pi.rho);
        if (use_negative_pressure_truncation)
            p = std::max(p, 0.0);
        pi.p = p;
    }
}

// compute boundary conditions for boundary particles
void Solver::compute_boundaryconditions()
{
// only loop over boundary particles
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < (int)boundary_indices.size(); ++idx)
    {
        int i = boundary_indices[idx];
        Particle &pi = particles[i];

        if (pi.vf.has_value())
        {
            (*pi.vf)[0] = 0.0;
            (*pi.vf)[1] = 0.0;
        }

        pi.p = 0.0;

        double vfx = 0.0, vfy = 0.0, pf = 0.0, Wi = 0.0, phx = 0.0, phy = 0.0;
        unsigned int neighbor_count = 0;

        for (int j : neighbors[i])
        {
            Particle &pj = particles[j];

            // skip other boundary particles
            if (pj.type == 1)
                continue;

            double dx = pi.x[0] - pj.x[0];
            double dy = pi.x[1] - pj.x[1];

            double r = min_image_dist(dx, dy, Lx, Ly);
            double Wij = kernel.getW(r, h);

            vfx -= pj.v[0] * Wij;
            vfy -= pj.v[1] * Wij;
            pf += pj.p * Wij;
            phx += pj.rho * dx * Wij;
            phy += pj.rho * dy * Wij;
            Wi += Wij;
            ++neighbor_count;
        }

        if (neighbor_count > 0 && Wi > 0.0)
        {
            vfx = vfx / Wi + 2.0 * pi.v[0];
            vfy = vfy / Wi + 2.0 * pi.v[1];
            // pf = pf / Wi + b[0] * (phx / Wi) + b[1] * (phy / Wi);
            pf = pf / Wi + this->b_eff[0] * (phx / Wi) + this->b_eff[1] * (phy / Wi);
            (*pi.vf)[0] = vfx;
            (*pi.vf)[1] = vfy;
            pi.p = pf;
            pi.rho = eos.density_from_pressure(pi.p);
        }
        else
        {
            (*pi.vf)[0] = 0.0;
            (*pi.vf)[1] = 0.0;
            pi.p = eos.get_bp();
            pi.rho = rho0;
        }
    }
}

// compute xsph velocity correction
void Solver::compute_xsph_velocity_correction()
{
// only loop over fluid particles
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < (int)fluid_indices.size(); ++idx)
    {
        int i = fluid_indices[idx];
        Particle &pi = particles[i];

        if (pi.v_xsph.has_value())
        {
            (*pi.v_xsph)[0] = 0.0;
            (*pi.v_xsph)[1] = 0.0;
        }

        double Wi = 0.0;

        double vx_xsph = 0.0;
        double vy_xsph = 0.0;

        unsigned int neighbor_count = 0;

        for (int j : neighbors[i])
        {
            Particle &pj = particles[j];

            double dx = pi.x[0] - pj.x[0];
            double dy = pi.x[1] - pj.x[1];

            double r = min_image_dist(dx, dy, Lx, Ly);
            double Wij = kernel.getW(r, h);

            double rho_j = pj.rho;
            vx_xsph += (pj.v[0] - pi.v[0]) * (pj.m / rho_j) * Wij;
            vy_xsph += (pj.v[1] - pi.v[1]) * (pj.m / rho_j) * Wij;

            Wi += Wij;
            ++neighbor_count;
        }

        if (neighbor_count > 0 && Wi > 0.0)
        {
            (*pi.v_xsph)[0] = pi.v[0] + eta * vx_xsph / Wi;
            (*pi.v_xsph)[1] = pi.v[1] + eta * vy_xsph / Wi;
        }
        else
        {
            (*pi.v_xsph)[0] = pi.v[0];
            (*pi.v_xsph)[1] = pi.v[1];
        }
    }
}

// compute particle forces (pressure ( + tensile instability correction ) ( + artificial viscosity ) ( + transport velocity) + viscous terms)
void Solver::compute_forces()
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)accel.size(); ++i)
        accel[i] = {0.0, 0.0};

    // only loop over fluid particles
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < (int)fluid_indices.size(); ++idx)
    {
        int i = fluid_indices[idx];
        Particle &pi = particles[i];

        double fx = 0.0, fy = 0.0;
        const double vx_i = pi.v[0];
        const double vy_i = pi.v[1];
        const double p_i = pi.p;
        const double rho_i = pi.rho;
        const double V_i = pi.m / rho_i;

        if (use_transport_velocity && pi.bpc.has_value())
        {
            (*pi.bpc)[0] = 0.0;
            (*pi.bpc)[1] = 0.0;
        }

        // only necessary for transport velocity calculation
        const double Ai11 = use_transport_velocity ? rho_i * vx_i * ((*pi.tv)[0] - vx_i) : 0.0;
        const double Ai12 = use_transport_velocity ? rho_i * vx_i * ((*pi.tv)[1] - vy_i) : 0.0;
        const double Ai21 = use_transport_velocity ? rho_i * vy_i * ((*pi.tv)[0] - vx_i) : 0.0;
        const double Ai22 = use_transport_velocity ? rho_i * vy_i * ((*pi.tv)[1] - vy_i) : 0.0;

        for (int j : neighbors[i])
        {
            Particle &pj = particles[j];

            const double vx_j = (pj.type == 1 && pj.vf.has_value()) ? (*pj.vf)[0] : pj.v[0];
            const double vy_j = (pj.type == 1 && pj.vf.has_value()) ? (*pj.vf)[1] : pj.v[1];
            const double p_j = pj.p;

            const double rho_j = pj.rho;
            const double V_j = pj.m / rho_j;

            double dx = pi.x[0] - pj.x[0];
            double dy = pi.x[1] - pj.x[1];
            double r = min_image_dist(dx, dy, Lx, Ly);
            double dW = kernel.getdW(r, h) / r;

            const double V_ij_sqr = V_i * V_i + V_j * V_j;

            // pressure force
            const double p_fac = V_ij_sqr * (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j);
            fx -= p_fac * dW * dx;
            fy -= p_fac * dW * dy;

            const double dvx = vx_i - vx_j;
            const double dvy = vy_i - vy_j;

            // using tensile instability correction if enabled
            if (use_tensile_instability_correction && pj.type == 0)
            {
                // tensile instability correction (Monaghan 2000)
                double Wij = kernel.getW(r, h);
                double W_dp = kernel.getW(dx0, h);
                double fij = Wij / W_dp;
                double fij_fourp = fij * fij * fij * fij;

                double tilde_pi = (p_i >= 0.0) ? 0.01 * p_i : epsilon * std::abs(p_i);
                double tilde_pj = (p_j >= 0.0) ? 0.01 * p_j : epsilon * std::abs(p_j);

                double tensile_p_fac = V_ij_sqr * (rho_j * tilde_pi + rho_i * tilde_pj) / (rho_i + rho_j);
                tensile_p_fac *= fij_fourp;

                fx -= tensile_p_fac * dW * dx;
                fy -= tensile_p_fac * dW * dy;
            }


            // using articial viscosity if enabled
            if (use_artificial_viscosity and pj.type == 0)
            {
                // avoid division by zero when r is extremely small
                const double r_sqr = r * r + 0.01 * h * h;

                // artificial viscosity (Monaghan 1992)
                double vr = dvx * dx + dvy * dy;
                if (vr < 0.0)
                {
                    double artvisc_fac = -(pi.m * pj.m * alpha * h * c * vr) / (((rho_i + rho_j) / 2) * r_sqr);
                    fx -= artvisc_fac * dW * dx;
                    fy -= artvisc_fac * dW * dy;
                }
            }

            // using transport velocity correction if enabled
            if (use_transport_velocity)
            {   
                const double Aj11 = (pj.type == 0) ? rho_j * vx_j * ((*pj.tv)[0] - vx_j) : 0.0;
                const double Aj12 = (pj.type == 0) ? rho_j * vx_j * ((*pj.tv)[1] - vy_j) : 0.0;
                const double Aj21 = (pj.type == 0) ? rho_j * vy_j * ((*pj.tv)[0] - vx_j) : 0.0;
                const double Aj22 = (pj.type == 0) ? rho_j * vy_j * ((*pj.tv)[1] - vy_j) : 0.0;

                const double tv_fac = 0.5 * V_ij_sqr;
                fx += tv_fac * ( (Ai11 + Aj11) * dW * dx + (Ai12 + Aj12) * dW * dy );
                fy += tv_fac * ( (Ai21 + Aj21) * dW * dx + (Ai22 + Aj22) * dW * dy );
                
                const double tv_bpc_fac = V_ij_sqr / pi.m; 
                (*pi.bpc)[0] -= tv_bpc_fac * eos.get_tvp_bp() * dW * dx;
                (*pi.bpc)[1] -= tv_bpc_fac * eos.get_tvp_bp() * dW * dy;
            }

            // viscous force
            const double visc_fac = V_ij_sqr * mu;
            fx += visc_fac * dW * dvx;
            fy += visc_fac * dW * dvy;
        }

        // compute acceleration and add (possibly damped) body force
        accel[i][0] = fx / pi.m + this->b_eff[0];
        accel[i][1] = fy / pi.m + this->b_eff[1];
    }
}
