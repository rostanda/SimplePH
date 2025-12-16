import SimplePH
import numpy as np
import math
import os
import pathlib
import sys

# threads for computing (default: 1)
num_threads = 1
if "-np" in sys.argv:
    idx = sys.argv.index("-np")
    num_threads = int(sys.argv[idx + 1])
SimplePH.set_omp_threads(num_threads)
print(f"OpenMP threads: {num_threads}")


def create_particles():
    parts = []
    for i in range(bcresx):
        for j in range(bcresy):
            p = SimplePH.Particle()

            # position
            x = -bcLx/2 + dx/2 + i * dx
            y = -bcLy/2 + dx/2 + j * dx
            p.x = [x, y]

            # velocity
            p.v = [0.0, 0.0]

            # mass and density
            rho = 1000.0
            p.m = rho * dx * dx
            p.rho = rho

            # default type: 0 = fluid
            p.type = 0

            # BC rundherum (außerhalb ursprünglicher Fluiddomain)
            if x < -Lx/2 or x > Lx/2 or y < -Ly/2 or y > Ly/2:
                p.type = 1

            # Alles, was Fluid ist aber rechts/oben liegt, als type=2 markieren
            if p.type == 0 and (x > 0.0 or y > 0.0):
                p.type = 2

            # Nur type < 2 hinzufügen (Fluid links unten + BC rundherum)
            if p.type < 2:
                parts.append(p)

    return parts


# Full fluid domain
Lx = 3.22     # x-width
Ly = 1.00     # y-height

# Fluid and solid subdomains
fluid_xmin = 0.744 + 1.248
fluid_zmax = 0.55
BOX_xmin = 0.744 - 0.5*0.161
BOX_xmax = 0.744 + 0.5*0.161
BOX_ymin = 0.295
BOX_ymax = 1.0 - 0.295
BOX_zmax = 0.161


# resolution in y-direction
res = 40 
dx = Ly / res
print("res:", res)
print("dx:", dx)

# resulting particles in x-direction
resy = res
resx = math.ceil(Lx / dx)

# kernel params
kappa = 2.0
h = 1.7 * dx
rcut = kappa * h

print("h:", h)
print("rcut:", rcut)

# boundary layer size
bcpartcount = math.ceil(rcut / dx)
bcresx = resx + 2 * bcpartcount
bcresy = resy + 2 * bcpartcount
bcLx = bcresx * dx
bcLy = bcresy * dx

print("bcpartcount:", bcpartcount)
print("bcresx:", bcresx)
print("bcresy:", bcresy)
print("bcLy:", bcLx)
print("bcLy:", bcLy)

# material params
rho = 971.0
V = dx * dx
m = rho * V
mu = 0.00971

print("m:", m)
print("V:", V)

# body force
b = [0.0, -9.81]

# reference values
vref = 1.0
Lref = 0.161

print("vref:", vref)

# Reynolds number
Re = (rho * vref * Lref) / mu
print("Re:", Re)

# density fluctuation
rho_fluct = 0.01

# solver setup
solver = SimplePH.Solver(
    h=h,
    Lx=bcLx,
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
solver.set_acceleration(b)

# compute wavespeed
solver.compute_soundspeed()

# compute timestep
solver.compute_timestep()

# equation of state
solver.set_eos(SimplePH.EOSType.Tait)

# set particles
solver.set_particles(create_particles())
# solver.set_particles(create_dambreak_particles())

# integrator
solver.set_integrator(SimplePH.VelocityVerletIntegrator())
# solver.set_integrator(SimplePH.EulerIntegrator())

# density method
solver.set_density_method(SimplePH.DensityMethod.Continuity)

# run simulation
output_name = "dambreak"

# create output directory and write VTU files into it
outdir = pathlib.Path(output_name)
outdir.mkdir(parents=True, exist_ok=True)

cwd = os.getcwd()
try:
    os.chdir(outdir)
    print(f"Writing VTU output into: {outdir.resolve()}")
    solver.run(10001, vtk_freq=100, log_freq=10)
finally:
    os.chdir(cwd)