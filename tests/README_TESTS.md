# SimplePH Unit Tests

This directory contains unit tests for SimplePH, covering solver initialization, material properties, and basic simulation execution.

## Running Tests

### Quick start (with dev environment)

If you use the project's development Python environment (example: `/home/user/simpleph`), run:

```bash
# Install dev dependencies into the venv
/home/user/simpleph/bin/python -m pip install -r ../requirements-dev.txt

# Run tests with pytest
/home/user/simpleph/bin/python -m pytest -v
```

Or activate the venv first:

```bash
source /home/user/simpleph/bin/activate
pip install -r ../requirements-dev.txt
pytest -v
```

### Running Tests

Standard Python test runner:

```bash
python3 test_simpleph.py
```

With pytest (recommended):

```bash
python3 -m pytest test_simpleph.py -v
```

Run a specific test class:

```bash
python3 -m pytest test_simpleph.py::TestSolverInitialization -v
```

Run a single test:

```bash
python3 -m pytest test_simpleph.py::TestMaterialProperties::test_set_viscosity -v
```

## Test Structure

### Solver Initialization (`TestSolverInitialization`)
- Create solvers with different kernel types (CubicSpline, WendlandC4)
- Verify solver object is properly initialized

### Material Properties (`TestMaterialProperties`)
- Viscosity, density, and acceleration setters
- Verify properties are correctly stored

### Physics Configuration (`TestPhysicsConfiguration`)
- EOS selection (Tait, Linear)
- Integrator selection (Euler, Velocity Verlet)
- Density method selection (Summation, Continuity)

### Particle Handling (`TestParticles`)
- Create fluid and boundary particles
- Set and retrieve particles from solver

### Stability Parameters (`TestStabilityParameters`)
- Soundspeed computation
- Timestep calculation for stability

### Simulation Execution (`TestSimulationExecution`)
- Run a minimal simulation with 3 particles
- Verify no runtime errors and particles are updated

## Notes

Tests focus on API correctness and robustness rather than physics accuracy. For validation of numerical methods, see `python/run_poiseuille_flow.py` which produces VTU outputs that can be compared against analytical solutions.
