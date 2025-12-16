import SimplePH
import numpy as np
import math
import os
import pathlib
import sys


# ----------------------------
# Thread count
# ----------------------------
num_threads = 1
if "-np" in sys.argv:
    idx = sys.argv.index("-np")
    num_threads = int(sys.argv[idx + 1])
SimplePH.set_omp_threads(num_threads)
print(f"OpenMP threads: {num_threads}")


# ============================================================
# SPHERIC TEST CASE 2 – 2D VERSION
# ============================================================

# Full domain from SPHERIC
LX = 3.22     # x-width
LY = 1.00     # y-height

# Characteristic scale
LREF = 0.161

# Resolution
res = 40
dx = LY / res
print("dx =", dx)

# Kernel
kappa = 2.0
h = 1.7 * dx
rcut = kappa * h
print("h =", h, "rcut =", rcut)

# Boundary cell count
bcpart = math.ceil(rcut / dx)

# Boundary-extended simulation domain
bcresx = math.ceil(LX / dx) + 2 * bcpart
bcresy = math.ceil(LY / dx) + 2 * bcpart

bcLx = bcresx * dx
bcLy = bcresy * dx
print("bc-domain:", bcLx, "x", bcLy)


# ============================================================
# Original SPHERIC geometry (2D projection)
# ============================================================

# Fluid block
FLD_xmin = (0.744 + 1.248)
FLD_zmax = 0.55          # becomes our *height*
FLD_ymin = 0.0
FLD_ymax = FLD_zmax

# Box
BOX_xmin = 0.744 - 0.5 * 0.161
BOX_xmax = 0.744 + 0.5 * 0.161
BOX_ymin = 0.295
BOX_ymax = 1.0 - 0.295      # symmetric in real 3D → we use 2D slice


# ============================================================
# PARTICLE GENERATOR
# ============================================================

def create_particles():
    parts = []

    for ix in range(bcresx):
        for iy in range(bcresy):

            p = SimplePH.Particle()

            x = -LX/2 - bcpart*dx + ix*dx
            y = -LY/2 - bcpart*dx + iy*dx

            p.x = [x, y]
            p.v = [0.0, 0.0]
            p.m = 1000.0 * dx * dx
            p.rho = 1000.0
            p.type = 1  # default: WALL

            # --------------------------
            # Detect fluid block
            # --------------------------
            if (FLD_xmin - LX/2 <= x <= LX/2
                and -LY/2 <= y <= FLD_ymax - LY/2):
                p.type = 0   # FLUID

            # --------------------------
            # Detect inside BOX (solid)
            # --------------------------
            if (BOX_xmin - LX/2 <= x <= BOX_xmax - LX/2
                and BOX_ymin - LY/2 <= y <= BOX_ymax - LY/2):
                p.type = 1   # SOLID

            parts.append(p)

    return parts


# ============================================================
# Solver Setup
# ============================================================

rho0 = 1000.0
rho_fluct = 0.01
mu = 0.00971

b = [0.0, -9.81]

# reference velocity (not super important)
vref = 1.0

solver = SimplePH.Solver(
    h=h,
    Lx=bcLx,
    Ly=bcLy,
    dx0=dx,
    Lref=LREF,
    vref=vref,
    kernel_type=SimplePH.KernelType.WendlandC4
)

solver.set_viscosity(mu)
solver.set_density(rho0, rho_fluct)
solver.set_acceleration(b)
solver.compute_soundspeed()
solver.compute_timestep()
solver.set_eos(SimplePH.EOSType.Tait)

print("Generating particles...")
solver.set_particles(create_particles())

solver.set_integrator(SimplePH.VelocityVerletIntegrator())
solver.set_density_method(SimplePH.DensityMethod.Continuity)


# ============================================================
# OUTPUT + RUN
# ============================================================

outdir = pathlib.Path("spheric_case2")
outdir.mkdir(exist_ok=True)
os.chdir(outdir)

solver.run(20000, vtk_freq=50, log_freq=20)
