# SimplePH – Minimal SPH Solver

A lightweight C++ implementation of a parallelized 2D weakly compressible Smoothed Particle Hydrodynamics (SPH) solver with Python bindings.

## Quick Start

### Build

```bash
mkdir -p build && cd build
cmake ..
make
```
The Python extension is placed in `python/SimplePH.so`.

### OpenMP Support

SimplePH uses OpenMP to parallelize computationally expensive parts of the solver.
OpenMP is enabled automatically when building the C++ code.

To ensure support on your system:

- **Linux (GCC/Clang)**:
```bash
sudo apt install libomp-dev
```
- **macOS (Clang via Homebrew):**:
```bash
brew install libomp
```
No Python dependencies are required for OpenMP.

### Run Example

```bash
cd python
python3 run_poiseuille_flow.py
```

Output VTU files are written to `poiseuille_flow/`.

### OpenMP Thread Control

The example script `run_poiseuille_flow.py` allows you to manually set the number of OpenMP threads:

```bash
python3 run_poiseuille_flow.py -np 8
```
Inside the script, OpenMP threads are configured via:

```bash
import SimplePH

num_threads = 1  # default

if "-np" in sys.argv:
    idx = sys.argv.index("-np")
    num_threads = int(sys.argv[idx + 1])

print(f"Setting {num_threads} OpenMP threads")
SimplePH.set_omp_threads(num_threads)
```

If no `-np` argument is given, the solver uses 1 thread.
To fully utilize your CPU, pass your number of physical cores.

### Plot Results

```bash
python3 plot_poiseuille_flow.py poiseuille_flow --out result.pdf
```

## Overview

SimplePH solves the incompressible Navier–Stokes equations in 2D using weakly compressible SPH (WC-SPH) discretization.

### Key Features

- **Simulation**: 2D fluid particles with pressure & viscous forces, periodic boundaries
- **Physics**: Configurable EOS (Tait, Linear) and density methods (Summation, Continuity)
- **Kernels**: Cubic Spline, Quintic Spline, Wendland C2 and Wendland C4 smoothing kernels
- **Integration**: Velocity Verlet and Euler time stepping schemes
- **Spatial Index**: Cell-linked neighbor search for efficiency
- **Visualization**: VTK output for ParaView, Python plotting utilities
- **Python API**: Full bindings via pybind11

### Particle Types

- **Fluid** (`type=0`): Evolves according to SPH dynamics
- **Boundary** (`type=1`): Fixed position; properties computed from neighboring fluid particles (ghost particles)

## Usage Example

```python
import SimplePH

# Create solver
solver = SimplePH.Solver(
    h=0.01, Lx=0.1, Ly=0.14, dx0=0.005,
    Lref=0.1, vref=0.006, kernel_type=SimplePH.KernelType.WendlandC4
)

# Material properties
solver.set_viscosity(mu=1.0)
solver.set_density(rho0=1000.0, rho_fluct=0.01)
solver.set_acceleration(b=[0.0024, 0.0])

# Physics configuration
solver.set_eos(SimplePH.EOSType.Tait)
solver.set_density_method(SimplePH.DensityMethod.Continuity)
solver.set_integrator(SimplePH.VelocityVerletIntegrator())

# Compute stability
solver.compute_soundspeed()
solver.compute_timestep()

# Create particles and simulate
particles = [...]  # SimplePH.Particle instances
solver.set_particles(particles)
solver.run(steps=100, vtk_freq=10, log_freq=10)
```

## Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Solver | `src/solver/` | Main SPH algorithm |
| Particle | `src/core/particle.hpp` | Particle data structure |
| Kernel | `src/sph/kernel.hpp` | Smoothing kernels and derivation (W, dW) |
| Grid | `src/neighbor/grid.hpp` | Cell-linked neighbor search |
| Integrators | `src/integrator/` | Time stepping (Euler, Velocity Verlet) |
| EOS | `src/sph/eos.hpp` | Pressure closure models |
| Bindings | `src/bindings.cpp` | Python interface (pybind11) |

## Testing

SimplePH includes a comprehensive test suite to ensure correctness, stability, and physical validity. The tests cover:

- **Unit Tests** – verify solver initialization, material property setters, particle creation, equation of state and integrator selection, and basic simulation execution.
- **Physics Invariants Tests** – validate mass conservation, density fluctuations, momentum symmetry, and periodic boundary handling.
- **Physical Flow Tests** – long-running convergence studies for standard flows (Poiseuille and Couette) with VTU output and L2-norm error comparison against analytical solutions.

### Running the Test Suite

Run all tests using pytest from the `tests/` directory:

```bash
cd tests
python3 -m pytest -v
```

## Performance

- Neighbor search: $O(n)$ per step via grid-based spatial indexing
- Force computation: $O(n \cdot k)$ where $k \approx 30$ dependent on chosen kernel (average neighbors)

## References

The SPH discretization follows standard formulations from:
- Monaghan, J. J. (1992). Smoothed particle hydrodynamics. In: Annual review of astronomy and astrophysics. Vol. 30 (A93-25826 09-90), p. 543-574., 30, 543-574.
- Monaghan, J. J. (2000). SPH without a tensile instability. Journal of computational physics, 159(2), 290-311.
- Monaghan, J. J. (2005). Smoothed particle hydrodynamics. Reports on progress in physics, 68(8), 1703.
- Hu, X. Y., & Adams, N. A. (2007). An incompressible multi-phase SPH method. Journal of computational physics, 227(1), 264-278.
- Adami, S., Hu, X. Y., & Adams, N. A. (2012). A generalized wall boundary condition for smoothed particle hydrodynamics. Journal of Computational Physics, 231(21), 7057-7075.

## License

MIT License. See `LICENSE` for details.

## Author

Daniel Rostan