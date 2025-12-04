#pragma once
#include <cmath>
#include <stdexcept>
#include <string>

// EOS types
enum class EOSType
{
    Tait,
    Linear
};

inline EOSType eos_from_string(const std::string &s)
{
    if (s == "Tait")
        return EOSType::Tait;
    if (s == "Linear")
        return EOSType::Linear;
    throw std::invalid_argument("Unknown EOS type: " + s);
}

// EOS class
class EOS
{
public:
    EOS(EOSType type, double rho0, double c, double bp = 0.0)
        : type(type), rho0(rho0), c(c), bp(bp) {}

    double pressure_from_density(double rho) const
    {
        switch (type)
        {
        case EOSType::Tait:
            return (c * c * rho0 / 7.0) * (std::pow(rho / rho0, 7.0) - 1.0) + bp;
        case EOSType::Linear:
            return (c * c) * (rho - rho0) + bp;
        default:
            throw std::logic_error("Unhandled EOS type");
        }
    }

    double density_from_pressure(double p) const
    {
        switch (type)
        {
        case EOSType::Tait:
            return rho0 * std::pow((p - bp) * (7.0 / (rho0 * c * c)) + 1, 1 / 7);
        case EOSType::Linear:
            return (p - bp) / (c * c) + rho0;
        default:
            throw std::logic_error("Unhandled EOS type");
        }
    }

private:
    EOSType type;
    double rho0;
    double c;
    double bp;
};
