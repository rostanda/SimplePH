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
            p.m = rho * dx * dx
            p.rho = rho

            # default type: 0 = fluid
            p.type = 0

            # BC rundherum (außerhalb ursprünglicher Fluiddomain)
            if x < -Lx/2 or x > Lx/2 or y < -Ly/2 or y > Ly/2:
                p.type = 1

            # Alles, was Fluid ist aber rechts/oben liegt, als type=2 markieren
            if p.type == 0 and (x > LxfluidBC or y > LyfluidBC):
                p.type = 2

            # Nur type < 2 hinzufügen (Fluid links unten + BC rundherum)
            if p.type < 2:
                parts.append(p)

    return parts


# Full domain
Lx = 5.366     # x-width
Ly = 2.0     # y-height

# Fluid subdomain size
Lxfluid = 2.0
Lyfluid = 1.0

# Fluid subdomain location from origin in domain middle
LxfluidBC = -0.683
LyfluidBC = 0.0

# resolution in y-direction
res = 80 
dx = Ly / res
print("res:", res)
print("dx:", dx)

# resulting particles in x-direction
resy = res
resx = math.ceil(Lx / dx)

# kernel params
kappa = 3.0
h = 1.3 * dx
rcut = kappa * h

print("h:", h)
print("rcut:", rcut)

# boundary layer size
bcpartcount = math.ceil(rcut / dx)
bcresx = resx + 4 * bcpartcount
bcresy = resy + 4 * bcpartcount
bcLx = bcresx * dx
bcLy = bcresy * dx

print("bcpartcount:", bcpartcount)
print("bcresx:", bcresx)
print("bcresy:", bcresy)
print("bcLy:", bcLx)
print("bcLy:", bcLy)

# material params
rho = 1.0
V = dx * dx
m = rho * V
mu = 0.0

print("m:", m)
print("V:", V)

# body force
b = [0.0, -1.0]

# reference values
vref = 2 * np.sqrt(1.0)
Lref = 1.0

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
    kernel_type=SimplePH.KernelType.QuinticSpline
)

# set viscosity
solver.set_viscosity(mu)

# set density params
solver.set_density(rho, rho_fluct)

# set body force
solver.set_acceleration(b,100)

# compute wavespeed
solver.compute_soundspeed()

# compute timestep
solver.compute_timestep()

# equation of state
solver.set_eos(SimplePH.EOSType.Tait,0.0)

# set particles
solver.set_particles(create_particles())
# solver.set_particles(create_dambreak_particles())

# integrator
solver.set_integrator(SimplePH.VelocityVerletIntegrator())
# solver.set_integrator(SimplePH.EulerIntegrator())

# density method
solver.set_density_method(SimplePH.DensityMethod.Continuity)

# use artificial viscosity
solver.activate_artificial_viscosity(True,0.4)

# run simulation
output_name = "dambreak_zhang2017"

# create output directory and write VTU files into it
outdir = pathlib.Path(output_name)
outdir.mkdir(parents=True, exist_ok=True)

cwd = os.getcwd()
try:
    os.chdir(outdir)
    print(f"Writing VTU output into: {outdir.resolve()}")
    solver.run(101, vtk_freq=1, log_freq=1)
finally:
    os.chdir(cwd)