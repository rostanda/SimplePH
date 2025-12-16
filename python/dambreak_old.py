#!/usr/bin/python
from __future__ import print_function, division
from hoosph   import *
from hoosph   import sph
from hoosph   import comm
from hoosph   import group
import numpy as np
import hoosph

# Create context
c = hoosph.context.initialize()

#
#  SPHERIC TEST CASE 2 : 
#  http://app.spheric-sph.org/sites/spheric/files/SPHERIC_Test2_v1p1.pdf
#  __________________________________
#  |                                |         1.0 m 
#  |                                |
#  |                          __    |
#  |                     _____\/____|        
#  |        0.161 m     |           |
#  |          _         |           |
#  |         | | 0.161 m|           | 0.55 m
#  |_________|_|________|___________|
#  |----> 0.744 m
#  |-------------------> 1.248 + 0.744 m
#             3.22 m

# System size
LX = 3.22 # m
LY = 1.00 # m
LZ = 1.00 # m

# Domain parameters
FLD_xmin = 0.744 + 1.248
FLD_zmax = 0.55
BOX_xmin = 0.744 - 0.5*0.161
BOX_xmax = 0.744 + 0.5*0.161
BOX_ymin = 0.295
BOX_ymax = 1.0 - 0.295
BOX_zmax = 0.161

# Characteristic scales
LREF = 0.161 # Box xwidth

# Discretization parameters
DX = FLD_zmax / 25.0           # m
V  = DX*DX*DX                  # m^3

# Fluid parameters
RHO0 =  971.0                  # kg / m^3
M    = RHO0*V                  # kg
DRHO = 0.01                    # %
MU   = 0.00971                 # Pa s

# SPH Parameters
KERNEL  = 'WendlandC2'
H       = hoosph.sph.kernel.OptimalH[KERNEL]*DX       # m
RCUT    = hoosph.sph.kernel.Kappa[KERNEL]*H           # m

# Physical parameters
GZ   = -9.81                    # m/s^2

# System size with ficticious wall particles
LXS = LX + 4*RCUT # m
LYS = LY + 4*RCUT # m
LZS = LZ + 4*RCUT # m


# Create initial particle distribution
unitcell = hoosph.lattice.sc(a=DX)
snapshot = unitcell.get_snapshot()
snapshot.replicate(int(LXS/DX),int(LYS/DX),int(LZS/DX))

if comm.get_rank() == 0:
    # Read particle data
    m   = snapshot.particles.mass[:]
    v   = snapshot.particles.velocity[:]
    x   = snapshot.particles.position[:]
    h   = snapshot.particles.slength[:]
    dpe = snapshot.particles.dpe[:]
    tid = snapshot.particles.typeid[:]
    
    # Set initial conditions
    snapshot.particles.types = ['F','S','D']
    
    for i in range(x.shape[0]):
        xi,yi,zi  = x[i][0], x[i][1], x[i][2]
        h[i]      = H
        dpe[i][0] = RHO0
        m[i]      = M
        v[i][0]   = 0.0
        v[i][1]   = 0.0
        v[i][2]   = 0.0
        tid[i]    = 2
        
        if ( xi < -0.5*LX or xi > 0.5*LX or
             zi < -0.5*LZ or zi > 0.5*LZ or
             yi < -0.5*LY or yi > 0.5*LY ):
            # Walls
            tid[i] = 1
        
        if ( xi > FLD_xmin-0.5*LX and xi < 0.5*LX ):
            if ( zi < FLD_zmax-0.5*LZ and zi > -0.5*LZ ):
                if ( yi > -0.5*LY and yi < 0.5*LY ):
                    # Fluid
                    tid[i] = 0
        
        if ( xi > BOX_xmin-0.5*LX and xi < BOX_xmax-0.5*LX ):
            if ( yi > BOX_ymin-0.5*LY and yi < BOX_ymax-0.5*LY ):
                if ( zi > -0.5*LZ and zi < BOX_zmax-0.5*LZ ):
                    # Box
                    tid[i] = 1
                
        snapshot.particles.velocity[:] = v
        snapshot.particles.mass[:]     = m
        snapshot.particles.slength[:]  = h
        snapshot.particles.dpe[:]      = dpe
        snapshot.particles.typeid[:]   = tid
    
# Set snapshot as system
system = hoosph.init.read_snapshot(snapshot)

# Define groups
groupFLUID  = group.type(name='FluidParticles',type='F')
groupSOLID  = group.type(name='SolidParticles',type='S')

# Kernel
Kernel = hoosph.sph.kernel.Kernels[KERNEL]()

# Neighbor list
NList = hoosph.nsearch.nlist.cell()
NList.set_params(r_buff=RCUT*0.05, check_period=1, kappa=Kernel.Kappa())

# Equation of State
EOS = hoosph.sph.eos.Tait()
EOS.set_params(RHO0,0.0)

# Set up SPH solver
model = hoosph.sph.models.SinglePhaseFlow(Kernel,EOS,NList,groupFLUID,groupSOLID)
model.densitymethod = 'CONTINUITY'
model.set_params(MU)
model.activateArtificialViscosity(0.2,0.0)
model.activateShepardRenormalization(20)
model.deactivateDensityDiffusion()
model.setBodyAcceleration(0.0,0.0,GZ,5000)
dt = model.compute_dt(LREF,0.0,DRHO)   

# Integrate
hoosph.sph.integrate.BaseIntegrator(dt=dt)
hoosph.sph.integrate.VelocityVerlet(group=groupFLUID)

# Dump trajectory
dump.gsd(filename="trajectory.gsd", overwrite=True, period=300, group=group.all(), static=[])
    
# Delete upper free surface
tags    = []
deleted = 0
for p in system.particles:
    if p.typeid == 2:
        tags.append(p.tag)
        deleted += 1
for t in tags:
    system.particles.remove(t)
if comm.get_rank() == 0: print("[INFO] {0} Particles were deleted.".format(deleted))

# Run
hoosph.run(50000)
