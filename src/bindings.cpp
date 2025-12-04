#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "solver.hpp"
#include "particle.hpp"
#include "integrator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(SimplePH, m)
{
    m.doc() = "SimplePH minimal SPH module";

    py::class_<Particle>(m, "Particle")
        .def(py::init<>())
        .def_readwrite("x", &Particle::x)
        .def_readwrite("v", &Particle::v)
        .def_readwrite("rho", &Particle::rho)
        .def_readwrite("p", &Particle::p)
        .def_readwrite("m", &Particle::m)
        .def_readwrite("type", &Particle::type);

    py::class_<Solver>(m, "Solver")
        .def(py::init<double, double, double, double, double, double, KernelType>(),
             py::arg("h"), py::arg("Lx"), py::arg("Ly"), py::arg("dx0"), py::arg("Lref"), py::arg("vref"), py::arg("kernel_type"))
        .def_readwrite("b", &Solver::b)
        .def_readwrite("mu", &Solver::mu)
        .def_readwrite("rho0", &Solver::rho0)
        .def_readwrite("rho_fluct", &Solver::rho_fluct)
        .def("set_particles", &Solver::set_particles)
        .def("set_acceleration", &Solver::set_acceleration,
             "Set constant acceleration vector b = [bx, by]")
        .def("set_viscosity", &Solver::set_viscosity)
        .def("set_viscosity", &Solver::set_viscosity,
             py::arg("mu"))
        .def("set_density", &Solver::set_density,
             py::arg("rho0"),
             py::arg("rho_fluct") = 0.01)
        .def("compute_soundspeed", &Solver::compute_soundspeed)
        .def("compute_timestep", &Solver::compute_timestep)
        .def("step", &Solver::step)
        .def("run", &Solver::run,
             py::arg("steps"),
             py::arg("vtk_freq") = 1,
             py::arg("log_freq") = 0)
        .def("set_eos", &Solver::set_eos,
             py::arg("eos_type"), py::arg("bp_fac") = 0.0)
        .def("set_density_method", &Solver::set_density_method)
        .def("set_particles", &Solver::set_particles)
        .def("set_integrator", &Solver::set_integrator)
        .def("get_particles", &Solver::get_particles,
             py::return_value_policy::reference_internal);

    py::enum_<KernelType>(m, "KernelType")
        .value("CubicSpline", KernelType::CubicSpline)
        .value("WendlandC4", KernelType::WendlandC4)
        .export_values();

    py::enum_<EOSType>(m, "EOSType")
        .value("Tait", EOSType::Tait)
        .value("Linear", EOSType::Linear)
        .export_values();

    py::enum_<Solver::DensityMethod>(m, "DensityMethod")
        .value("Summation", Solver::DensityMethod::Summation)
        .value("Continuity", Solver::DensityMethod::Continuity)
        .export_values();

    py::class_<Integrator, std::shared_ptr<Integrator>>(m, "Integrator");

    py::class_<EulerIntegrator, Integrator, std::shared_ptr<EulerIntegrator>>(m, "EulerIntegrator")
        .def(py::init<>());

    py::class_<VelocityVerletIntegrator, Integrator, std::shared_ptr<VelocityVerletIntegrator>>(m, "VelocityVerletIntegrator")
        .def(py::init<>());
}
