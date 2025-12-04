#pragma once

#include <vector>
#include <string>
#include <optional>
#include <array>
#include <ostream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cmath>

#include "particle.hpp"

class VTKWriter
{
public:
    VTKWriter() = default;
    ~VTKWriter() = default;

    // static function
    static void write(const std::vector<Particle> &particles, int step);

    // write to custom filename
    static void write(const std::vector<Particle> &particles, int step, const std::string &filename);
};
