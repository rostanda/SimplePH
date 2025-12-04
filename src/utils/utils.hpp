#pragma once
#include <cmath>

inline double min_image_dist(double &dx, double &dy, double Lx, double Ly)
{
    if (dx > 0.5 * Lx)
        dx -= Lx;
    if (dx < -0.5 * Lx)
        dx += Lx;
    if (dy > 0.5 * Ly)
        dy -= Ly;
    if (dy < -0.5 * Ly)
        dy += Ly;
    return std::sqrt(dx * dx + dy * dy);
}