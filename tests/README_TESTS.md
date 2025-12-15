# SimplePH Unit and Physical Flow Tests

This directory contains all unit tests, physics invariants tests, and physical flow tests for SimplePH. It covers:


- Solver initialization
- Material properties
- Particle handling
- Physics configuration
- Stability parameter computation
- Minimal simulation execution
- Physics invariants validation
- Physical flow validation (Poiseuille and Couette flow) with VTU outputs

## Running Tests

### Quick start

Install development dependencies (only needed once) for pytest:


```bash
pip install -r ../requirements-dev.txt
```
Run all tests with verbose output and pytest:

```
pytest -v
```

### Running API/Unit Tests

Standard Python test runner:

```bash
python3 test_api.py
```

With pytest (recommended):

```bash
python3 -m pytest test_api.py -v
```

Run a specific test class:

```bash
python3 -m pytest test_api.py::TestSolverInitialization -v
```

Run a single test method:

```bash
python3 -m pytest test_api.py::TestMaterialProperties::test_set_viscosity -v
```

### Running Physical Invariant Tests

Run all physical invarian tests:

```bash
python3 -m pytest test_invariants.py -v
```

### Running Physical Flow Tests

Physical flow tests include:

- Poiseuille flow: convergence study for multiple resolutions (10, 20, 40)

- Couette flow: convergence study for multiple resolutions (10, 20, 40)

Run all physical flow tests:

```bash
python3 -m pytest test_physical_flows.py -v
```

VTU outputs:
    - Poiseuille: `poiseuille_<res>/`
    - Couette: `couette_<res>/`

These directories are preserved after test runs for inspection.

### Optional: Direct Execution

You can also run the tests directly via Python without pytest:

```bash
python3 test_api.py
python3 test_invariants.py
python3 test_physical_flows.py
```

In `test_physical_flows.py`, plots are automatically generated after running the flow tests.

## Test Structure

### Unit Tests

#### Solver Initialization (`TestSolverInitialization`)
- Create solvers with different kernel types (CubicSpline, WendlandC4)
- Verify solver object is properly initialized

#### Material Properties (`TestMaterialProperties`)
- Viscosity, density, and acceleration setters
- Verify properties are correctly stored

#### Physics Configuration (`TestPhysicsConfiguration`)
- EOS selection (Tait, Linear)
- Integrator selection (Euler, Velocity Verlet)
- Density method selection (Summation, Continuity)

#### Particle Handling (`TestParticles`)
- Create fluid and boundary particles
- Set and retrieve particles from solver

#### Stability Parameters (`TestStabilityParameters`)
- Soundspeed computation
- Timestep calculation for stability

#### Advanced Solver Features (`TestAdvancedSolverFeatures`)
- Artificial viscosity
- Tensile instability correction

#### Simulation Execution (`TestSimulationExecution`)
- Run a minimal simulation with 3 particles
- Verify no runtime errors and particles are updated

### Physic Invariant Tests
- Mass Conservation:
    - Verifies that total particle mass remains constant over time
    - Independent of kernel choice and density formulation
- Density Fluctuations:
    - Checks that relative density deviations remain below the prescribed `rho_fluct`
    - Ensures consistency of EOS, density update, and time integration
- Pairwise Force Symmetry:
    - Confirms antisymmetry of pairwise particle interactions
    - Ensures internal forces do not generate spurious net momentum
- Velocity Symmetry:
    - For a homogeneous, force-free periodic system, total momentum must remain zero
    - Detects broken kernel gradient symmetry or integration errors
- Periodic Boundary Consistency:
    - Verifies that particle positions remain within the periodic domain
    - Ensures correct wrapping and neighbor search across domain boundaries
- Parameter Coverage (Each invariant is tested for all combinations of):
    - Kernels
    - Density Methods

### Physical Flow Tests (test_physical_flows.py)
- Poiseuille Flow:
    - Convergence study with multiple resolutions
    - L2-norm error computed against analytical parabolic solution
    - VTU outputs for post-processing

- Couette Flow:
    - Convergence study with multiple resolutions
    - L2-norm error computed against analytical parabolic solution
    - VTU outputs for post-processing

- Plot Generation:
    - A dedicated test `test_plot_results()` generates and saves convergence plots automatically
    - Can be called directly or via pytest

## Notes
- Unit tests focus on API correctness and robustness rather than physics accuracy
- Physical flow tests are longer-running and produce VTU outputs that can be compared against analytical or expected solutions
- VTU directories are preserved after test runs to allow inspection and plotting of flow fields