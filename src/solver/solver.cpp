#include <cstdio>
#include <algorithm>
#include <cmath>
#include <limits>
#include "solver.hpp"
#include "kernel.hpp"
#include "utils.hpp"
#include "integrator.hpp"

// constructor
Solver::Solver(double h_, double Lx_, double Ly_, double dx0_, double Lref_, double vref_, KernelType kernel_type_)
    : h(h_), Lx(Lx_), Ly(Ly_), dx0(dx0_), Lref(Lref_), vref(vref_), kernel(kernel_type_, h_),
      grid(kernel.get_rcut(h_), Lx_, Ly_)
{
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
        {
            p.vf = {0.0, 0.0};
            p.pf = 0.0;
        }
        else
        {
            p.vf = std::nullopt;
            p.pf = std::nullopt;
        }
    }

    for (auto &p : particles)
    {
        if (density_method == DensityMethod::Summation)
            p.drho_dt = std::nullopt;
        else if (density_method == DensityMethod::Continuity)
            p.drho_dt = 0.0;
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
void Solver::set_acceleration(const std::array<double, 2> &b_) { b = b_; }

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

// set EOS
void Solver::set_eos(EOSType eos_type_, double bp_) { eos = EOS(eos_type_, rho0, c, bp_); }

// get particles
const std::vector<Particle> &Solver::get_particles() const { return particles; }

// solve one timestep
void Solver::step()
{
    if (density_method == DensityMethod::Continuity)
        compute_density_continuity();
    integrator->step1(particles, fluid_indices, accel, dt, Lx, Ly);
    update_neighbors();
    if (density_method == DensityMethod::Summation)
        compute_density_summation();
    compute_pressure();
    compute_boundaryconditions();
    compute_forces();
    integrator->step2(particles, fluid_indices, accel, dt, Lx, Ly);
}

// run simulation
void Solver::run(int steps, int vtk_freq, int log_freq)
{
    if (vtk_freq <= 0)
        vtk_freq = 1;
    if (log_freq < 0)
        log_freq = 0;

    int vtk_counter = 0, log_counter = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < steps; ++i)
    {
        auto step_start = std::chrono::high_resolution_clock::now();

        if (vtk_counter == 0)
            VTKWriter::write(particles, i);
        vtk_counter = (vtk_counter + 1) % vtk_freq;
        if (log_freq > 0 && log_counter == 0)
            std::cout << "Step " << i << std::endl;
        if (log_freq > 0)
            log_counter = (log_counter + 1) % log_freq;
        step();

        auto step_end = std::chrono::high_resolution_clock::now();
        double step_time = std::chrono::duration<double>(step_end - step_start).count();

        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - start_time).count();

        if (log_freq > 0 && log_counter == 1)
            std::cout << "  Iteration time: " << step_time << "s | Total elapsed: " << elapsed << "s" << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    total_elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "\n=== Simulation completed ===" << std::endl;
    std::cout << "Total time: " << total_elapsed_time << "s" << std::endl;
    std::cout << "Total steps: " << steps << std::endl;
    std::cout << "Average time per step: " << (total_elapsed_time / steps) << "s" << std::endl;
}

// update neighbor list
void Solver::update_neighbors()
{
    neighbors.clear();
    neighbors.resize(particles.size());
    grid.build(particles);
    const std::size_t n = particles.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        // grid.find_neighbors calls the provided lambda for each neighbor pair
        grid.find_neighbors(static_cast<int>(i), particles,
                            [&](int pid, int jid, double dx, double dy, double r)
                            { neighbors[pid].push_back(jid); });
    }
}

// compute density summation (rho_i = sum_j m_j W(r_ij))
void Solver::compute_density_summation()
{
    // only loop over fluid particles
    for (int i : fluid_indices)
    {
        Particle &pi = particles[i];
        double rho = 0.0;

        for (int j : neighbors[i])
        {
            Particle &pj = particles[j];
            double dx = pj.x[0] - pi.x[0];
            double dy = pj.x[1] - pi.x[1];
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
    for (auto &p : particles)
        p.drho_dt = 0.0;
    
    // only loop over fluid particles
    for (int i : fluid_indices)
    {
        Particle &pi = particles[i];

        double drho_dt_i = 0.0;
        const double vi_x = pi.v[0];
        const double vi_y = pi.v[1];
        const double rho_i = pi.rho;

        for (int j : neighbors[i])
        {
            Particle &pj = particles[j];
            if (pj.type == 1)
                continue;

            double dx = pi.x[0] - pj.x[0];
            double dy = pi.x[1] - pj.x[1];
            double r = min_image_dist(dx, dy, Lx, Ly);
            double dW = kernel.getdW(r, h) / r;

            const double dvx = vi_x - pj.v[0];
            const double dvy = vi_y - pj.v[1];

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
    
    // only loop pver fluid particles
    for (int i : fluid_indices)
    {
        Particle &pi = particles[i];
        pi.p = eos.pressure_from_density(pi.rho);
    }
}

// compute boundary conditions for boundary particles
void Solver::compute_boundaryconditions()
{
    for (int i : boundary_indices)
    {
        Particle &pi = particles[i];

        if (!pi.vf.has_value())
            pi.vf = {0.0, 0.0};
        if (!pi.pf.has_value())
            pi.pf = 0.0;

        double vfx = 0.0, vfy = 0.0, pf = 0.0, Wi = 0.0, phx = 0.0, phy = 0.0;
        unsigned int neighbor_count = 0;

        for (int j : neighbors[i])
        {
            Particle &pj = particles[j];

            // skip other boundary particles
            if (pj.type == 1)
                continue;

            double dx = pj.x[0] - pi.x[0];
            double dy = pj.x[1] - pi.x[1];
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
            pf = pf / Wi + b[0] * (phx / Wi) + b[1] * (phy / Wi);
            pi.rho = eos.density_from_pressure(pi.p);
        }
        else
        {
            pi.rho = rho0;
        }

        (*pi.vf)[0] = vfx;
        (*pi.vf)[1] = vfy;
        *pi.pf = pf;
    }
}

// compute particle forces (pressure + viscous terms)
void Solver::compute_forces()
{
    accel.assign(accel.size(), {0.0, 0.0});
    const std::size_t n = particles.size();

    // only loop pver fluid particles
    for (int i : fluid_indices)
    {
        Particle &pi = particles[i];

        double fx = 0.0, fy = 0.0;
        const double vi_x = pi.v[0];
        const double vi_y = pi.v[1];
        const double p_i = pi.p;
        const double rho_i = pi.rho;
        const double Vi = pi.m / rho_i;

        for (int j : neighbors[i])
        {
            Particle &pj = particles[j];

            const double vj_x = (pj.type == 1 && pj.vf.has_value()) ? (*pj.vf)[0] : pj.v[0];
            const double vj_y = (pj.type == 1 && pj.vf.has_value()) ? (*pj.vf)[1] : pj.v[1];
            const double p_j = (pj.type == 1 && pj.pf.has_value()) ? *pj.pf : pj.p;
            const double rho_j = pj.rho;
            const double Vj = pj.m / rho_j;

            double dx = pi.x[0] - pj.x[0];
            double dy = pi.x[1] - pj.x[1];
            double r = min_image_dist(dx, dy, Lx, Ly);
            double dW = kernel.getdW(r, h) / r;

            const double Vij_sqr = Vi * Vi + Vj * Vj;
            const double p_fac = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j);

            fx -= Vij_sqr * p_fac * dW * dx;
            fy -= Vij_sqr * p_fac * dW * dy;

            const double dvx = vi_x - vj_x;
            const double dvy = vi_y - vj_y;
            const double visc_fac = dW * dx * dx + dW * dy * dy;

            // avoid division by zero when r is extremely small
            const double r2 = (r > 0.0) ? (r * r) : 1e-12;
            fx += (mu * Vij_sqr * visc_fac / r2) * dvx;
            fy += (mu * Vij_sqr * visc_fac / r2) * dvy;
        }

        // compute acceleration and add body force
        accel[i][0] = fx / pi.m + b[0];
        accel[i][1] = fy / pi.m + b[1];
    }
}
