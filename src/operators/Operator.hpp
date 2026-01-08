#pragma once

// Forward declaration
struct SolverState;

class Operator {
public:
    virtual ~Operator() = default;
    virtual void apply(SolverState& state) = 0;
};