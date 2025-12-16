# SimplePH AI Coding Agent Instructions

## Project Overview

**SimplePH** is a minimal Smoothed Particle Hydrodynamics (SPH) solver written in modern C++ and exposed to Python via pybind11. It simulates 2D fluid dynamics (e.g., channel flow) with particles, boundary conditions, and configurable physics parameters.

## Architecture & Key Components

### Core Simulation Loop (`src/solver/solver.hpp`, `solver.cpp`)

The `Solver` class orchestrates the SPH simulation:

1. **Initialization**: Takes kernel type (`CubicSpline` or `WendlandC4`), domain dimensions (`Lx`, `Ly`), and smoothing length `h`
2. **Per-timestep** (`step()`):
   - Update neighbor list via spatial grid
   - Compute particle densities (Summation or Continuity method)
   - Calculate pressure from EOS
   - Apply boundary conditions (BC particles have `type=1`)
   - Compute forces and accelerations
   - Integrate using selected integrator (Euler or Velocity Verlet)

**Key insight**: BC particles (walls) are ghost particles with fictitious velocity/pressure computed from neighboring fluid particles—not rigid constraints.

### Particle System (`src/core/particle.hpp`)

Particles have type-specific behavior:
- **Fluid** (`type=0`): All standard properties (x, v, rho, p, m)
- **Boundary** (`type=1`): Fixed position; `vf` and `pf` are computed, not prescribed

Optional fields (`std::optional`) exist for:
- `drho_dt`: Only populated if using Continuity density method
- `mui`, `vf`, `pf`: Support non-Newtonian fluids and BC (initialize in `set_particles()`)

### SPH Physics

**Kernels** (`src/sph/kernel.hpp`): Two implementations with matching cutoff radius ($2h$):
- **CubicSpline**: Faster, older standard
- **WendlandC4**: Smoother, higher-order accuracy

Kernels use normalized forms; always call via `Kernel::getW()` and `Kernel::getdW()` methods, never inline.

**Density Methods** (`Solver::DensityMethod`):
- **Summation** ($\rho_i = \sum_j m_j W(r_{ij})$): Direct but noisy
- **Continuity** ($d\rho_i/dt = ...$): Smoother; requires `drho_dt` initialization

**EOS** (`src/sph/eos.hpp`): Tait or Linear; pressure computed from density deviation.

### Spatial Indexing (`src/neighbor/grid.hpp`)

Grid-based neighbor search with periodic boundary conditions (min-image convention):
- Cell resolution auto-computed: `n_cells = floor(L / rcut)`, adjusted to ensure `cell_size ≥ rcut`
- Periodic wrapping handled by `Grid::find_neighbors()` callback and `min_image_dist()` utility
- Neighbor list built each step; no persistent data structure

### Integrators (`src/integrator/`)

Abstract `Integrator` base class with two implementations:
- **Euler**: Simple forward step; less stable
- **VelocityVerlet**: Two-step scheme (`step1`, `step2`); more stable for fluid dynamics

Both handle periodic boundaries internally. Switch via `Solver::set_integrator()`.

## Python Interface & Build Workflow

### pybind11 Bindings (`src/bindings.cpp`)

Exposes C++ classes/enums directly to Python:
- `Solver`, `Particle`, `Integrator` hierarchy, `KernelType`, `EOSType`, `DensityMethod`
- Particle fields are read-write; can modify in Python after `get_particles()`
- `get_particles()` returns reference—modifications affect solver state

### Build & Setup

**CMake structure**:
```
src/CMakeLists.txt: Compiles sph_core library + Python extension SimplePH
→ Finds pybind11 via: python3 -m pybind11 --cmakedir
→ Output: python/SimplePH.so (imported as 'import SimplePH')
```

**Build command** (from workspace root):
```bash
mkdir -p build && cd build
cmake ..
make
# Produces: python/SimplePH.so
```

**Python workflow**:
```python
import SimplePH
solver = SimplePH.Solver(h=0.01, Lx=0.1, Ly=0.1, ...)
solver.set_particles([...])
solver.run(steps=100, vtk_freq=10, log_freq=0)
```

See `python/run_channel_flow.py` for reference: Generates periodic BC particles, sets material params (rho, mu, gravity), computes reference velocity and sound speed, then runs.

## Development Conventions & Patterns

### Particle State Management

- **Never** modify `particle.type` after `set_particles()`; type determines BC vs fluid logic
- Initialize optional fields: if using Continuity method, `drho_dt` must be `0.0` or `std::nullopt` before stepping
- BC particle fictitious values must be computed in `compute_boundaryconditions()` before force calculation

### Kernel & Neighbor Distance

- **Always** normalize kernel gradients: use `getdW(r, h) / r` (already done in code)
- **Always** use `min_image_dist()` for pair distances to apply periodic wrapping
- Cutoff is implicit in kernel (`dW=0` beyond $2h$); no explicit pruning needed

### Numerical Stability

- Sound speed computed as $c = \max(c_{CFL}, c_{GW}, c_{FC})$ from CFL, gravity-wave, and viscous Fourier conditions
- Timestep is the minimum of three stability limits—solver prints all candidates
- Density fluctuation (`rho_fluct`) normalizes pressure; larger values → more compressible fluid

### Code Locations for Common Tasks

| Task | File(s) |
|------|---------|
| Add new kernel | `src/sph/kernel.hpp`: Add inline functions + enum case |
| Modify force law | `Solver::compute_forces()` in `solver.cpp` |
| Change BC logic | `Solver::compute_boundaryconditions()` in `solver.cpp` |
| Add particle property | `Particle` struct + pybind11 binding in `bindings.cpp` |
| Implement new integrator | New class inheriting `Integrator` (see `euler_integrator.hpp`) |

## Debugging & Output

- **VTK output**: Called at intervals; written to `particles_XXXXX.vtu` in working directory
- **Neighbor counts**: Call `print_neighbor_counts(solver)` (declared in `solver.hpp`, defined in `debug_utils.hpp`)
- **Print statements**: Use `printf()` in C++, Python's `print()` in simulation scripts
- **Grid layout**: Grid constructor prints cell info and neighbor topology on initialization

## Typical Workflow for Modifications

1. **Physics changes**: Edit `Solver::compute_forces()` or density methods
2. **Particle properties**: Modify `Particle` struct + pybind11 binding + optional initialization in `set_particles()`
3. **Kernel/EOS**: Add enum variants + implementations in corresponding hpp files
4. **Integration schemes**: New subclass of `Integrator` with `step1()`/`step2()` overrides
5. **Build & test**: `cmake .. && make` then run `python/run_channel_flow.py` or custom script
