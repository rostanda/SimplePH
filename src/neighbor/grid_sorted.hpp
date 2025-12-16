#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <unordered_set>
#include <cstdio>
#include "particle.hpp"
#include "utils.hpp"

class Grid
{
public:
    Grid(double rcut_, double Lx_, double Ly_)
        : rcut(rcut_), Lx(Lx_), Ly(Ly_)
    {
        nx = static_cast<int>(std::floor(Lx / rcut));
        ny = static_cast<int>(std::floor(Ly / rcut));
        if (nx < 1)
            nx = 1;
        if (ny < 1)
            ny = 1;

        hx = Lx / nx;
        hy = Ly / ny;

        while (nx > 1 && hx < rcut)
        {
            nx--;
            hx = Lx / nx;
        }
        while (ny > 1 && hy < rcut)
        {
            ny--;
            hy = Ly / ny;
        }

        cells.resize(nx * ny);
        cell_neighbors.resize(nx * ny);

        // fill neighbors
        for (int cy = 0; cy < ny; ++cy)
        {
            for (int cx = 0; cx < nx; ++cx)
            {
                int idx = cy * nx + cx;
                int k = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        int ncx = (cx + dx + nx) % nx;
                        int ncy = (cy + dy + ny) % ny;
                        cell_neighbors[idx][k++] = ncy * nx + ncx;
                    }
            }
        }
    }

    // clear cell lists
    void clear()
    {
        for (auto &c : cells)
            c.clear();
    }

    // BUILD GRID + BUILD SORT
    void build(const std::vector<Particle> &particles)
    {
        clear();

        const int N = particles.size();
        sorted_ids.clear();
        sorted_ids.reserve(N);

        // build cell lists
        for (int i = 0; i < N; i++)
        {
            double xp = fmod(particles[i].x[0], Lx);
            if (xp < 0)
                xp += Lx;

            double yp = fmod(particles[i].x[1], Ly);
            if (yp < 0)
                yp += Ly;

            int cx = int(xp / hx);
            int cy = int(yp / hy);

            cells[cy * nx + cx].push_back(i);
        }

        // now build sorted order by concatenating cells
        for (auto &cell : cells)
            for (int pid : cell)
                sorted_ids.push_back(pid);

        // build inverse map
        inv_id.resize(N);
        for (int newpos = 0; newpos < N; newpos++)
        {
            inv_id[sorted_ids[newpos]] = newpos;
        }
    }

    // neighbor search in original particle array
    template <typename Func>
    void find_neighbors(int pid,
                        const std::vector<Particle> &particles,
                        Func callback) const
    {
        const Particle &p = particles[pid];

        double xp = fmod(p.x[0], Lx);
        if (xp < 0)
            xp += Lx;
        double yp = fmod(p.x[1], Ly);
        if (yp < 0)
            yp += Ly;

        int cx = int(xp / hx);
        int cy = int(yp / hy);

        for (int nb : cell_neighbors[cy * nx + cx])
        {
            const auto &cell = cells[nb];

            for (int j : cell)
            {
                if (j == pid)
                    continue;

                double dx = particles[j].x[0] - p.x[0];
                double dy = particles[j].x[1] - p.x[1];
                double r = min_image_dist(dx, dy, Lx, Ly);

                if (r <= rcut)
                    callback(pid, j, dx, dy, r);
            }
        }
    }

public:
    // additional data structures for sorted GRID layout
    std::vector<int> sorted_ids; // particle ids sorted by cell
    std::vector<int> inv_id;     // inverse map

private:
    double rcut, Lx, Ly;
    int nx, ny;
    double hx, hy;

    std::vector<std::vector<int>> cells;
    std::vector<std::array<int, 9>> cell_neighbors;
};
