import SimplePH
import numpy as np
import math
import os
import pathlib

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

print

# overall geometry
Lx = 0.2
Ly = 0.1

# fluid 


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
rho = 1000.0
V = dx * dx
m = rho * V
mu = 1.0

print("m:", m)
print("V:", V)

# body force
b = [0.0, -0.01]

# reference values
vmax = b[0] * Lx**2 * 0.25 / (mu / rho)
# vref = (2/3) * vmax
# Lref = Lx / 2
vref = vmax
Lref = Ly

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