#pragma once
#include "Operator.hpp"

class PressureViscousForce : public Operator {
public:
    void apply(SolverState& state) override;
};
