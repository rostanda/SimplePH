import SimplePH
import numpy as np
import math
import os
import pathlib
import sys
import argparse

parser = argparse.ArgumentParser(description="SimplePH couette flow")

parser.add_argument(
    "-np", "--num-threads",
    type=int,
    default=1,
    help="Number of OpenMP threads"
)

parser.add_argument(
    "-res", "--resolution",
    type=int,
    default=40,
    help="Resolution"
)

args = parser.parse_args()

num_threads = args.num_threads

print(f"Setting {num_threads} OpenMP threads")
SimplePH.set_omp_threads(num_threads)

# create particle layout
def create_particles():
    parts = []
    for i in range(res):
        for j in range(bcresy):
            p = SimplePH.Particle()

            # set position
            p.x = [
                -Lx/2 + dx/2 + i * dx,
                -bcLy/2 + dx/2 + j * dx
            ]

            # init velocity
            p.v = [0.0, 0.0]

            # mass and density
            p.m = m
            p.rho = rho

            # set type
            p.type = 0
            if p.x[1] > Ly/2 or p.x[1] < -Ly/2:
                p.type = 1
            # set wall velocity of upper wall
            if p.x[1] > Ly/2:
                p.v = [vwall, 0.0]

            parts.append(p)
    return parts


# geometry
Lx = 0.1
Ly = 0.1

# resolution
res = args.resolution
dx = Lx / res

print("res:", res)
print("dx:", dx)

# kernel params
kappa = 2.0
h = 1.7 * dx
rcut = kappa * h

print("h:", h)
print("rcut:", rcut)

# boundary layer size
bcpartcount = math.ceil(rcut / dx)
bcresy = 2 * bcpartcount + res
bcLy = bcresy * dx

print("bcpartcount:", bcpartcount)
print("bcresy:", bcresy)
print("bcLy:", bcLy)

# material params
rho = 1000.0
V = dx * dx
m = rho * V
mu = 1.0

print("m:", m)
print("V:", V)

# body force
b = [0.0, 0.0]

# upper wall velocity
vwall = 0.002

# reference values
vmax = vwall
vref = vmax
Lref = Ly

print("vref:", vref)

# characteristic velocity and length (mean velocity and channel height)
vchar = vmax / 2.0
Lchar = Ly
# Reynolds number
Re = (rho * vchar * Lchar) / mu
print("Re:", Re)

# density fluctuation
rho_fluct = 0.01

# solver setup
solver = SimplePH.Solver(
    h=h,
    Lx=Lx,
    Ly=bcLy,
    dx0=dx,
    Lref=Lref,
    vref=vref,
    kernel_type=SimplePH.KernelType.WendlandC4
)

# set viscosity
solver.set_viscosity(mu)

# set density params
solver.set_density(rho, rho_fluct)

# set body force
solver.set_acceleration(b,0)

# compute wavespeed
solver.compute_soundspeed()

# compute timestep
solver.compute_timestep()

# equation of state
solver.set_eos(SimplePH.EOSType.Tait)

# use artificial viscosity
# solver.activate_artificial_viscosity(0.01)

# use tensile instability correction
# solver.activate_tensile_instability_correction()

# # use xsph filter
# solver.activate_xsph_filter(0.1)

# # use transport velocity formulation
# solver.activate_transport_velocity()

# set particles
solver.set_particles(create_particles())

# integrator
solver.set_integrator(SimplePH.VerletIntegrator())
# solver.set_integrator(SimplePH.EulerIntegrator())
# solver.set_integrator(SimplePH.TransportVelocityVerletIntegrator())

# density method
# solver.set_density_method(SimplePH.DensityMethod.Continuity)
solver.set_density_method(SimplePH.DensityMethod.Summation)

# set output name
output_name = "couette_flow"
solver.set_output_name(output_name)

# run simulation
solver.run(5001, vtk_freq=100, log_freq=50)
