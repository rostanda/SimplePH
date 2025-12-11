import SimplePH
import numpy as np
import math
import os
import pathlib
import sys

# threads for computing (default: 1)
num_threads = 1

# override via -np <N> as command line argument
if "-np" in sys.argv:
    idx = sys.argv.index("-np")
    try:
        num_threads = int(sys.argv[idx + 1])
    except (IndexError, ValueError):
        raise ValueError("Usage: script.py [-np N]")

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

            parts.append(p)
    return parts

# geometry
Lx = 0.1
Ly = 0.1

# resolution
res = 40
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
b = [0.0024, 0.0]

# reference values
vmax = (rho * b[0] * (Ly/2)**2) / (2 * mu)
vref = vmax
Lref = Ly

print("vref:", vref)

# Reynolds number
vchar = (2/3) * vmax
Lchar = Ly / 2
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

# set particles
solver.set_particles(create_particles())

# integrator
solver.set_integrator(SimplePH.VelocityVerletIntegrator())
# solver.set_integrator(SimplePH.EulerIntegrator())

# density method
# solver.set_density_method(SimplePH.DensityMethod.Continuity)
solver.set_density_method(SimplePH.DensityMethod.Summation)

# run simulation
output_name = "poiseuille_flow"

# create output directory and write VTU files into it
outdir = pathlib.Path(output_name)
outdir.mkdir(parents=True, exist_ok=True)

cwd = os.getcwd()
try:
    os.chdir(outdir)
    print(f"Writing VTU output into: {outdir.resolve()}")
    solver.run(5001, vtk_freq=100, log_freq=50)
finally:
    os.chdir(cwd)