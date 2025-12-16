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
        // number cells
        nx = static_cast<int>(std::floor(Lx / rcut));
        ny = static_cast<int>(std::floor(Ly / rcut));
        if (nx < 1)
            nx = 1;
        if (ny < 1)
            ny = 1;

        hx = Lx / nx;
        hy = Ly / ny;

        // decrease cell numbers, if hx or hy < rcut
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

        printf("Grid init: nx=%d ny=%d hx=%.3f hy=%.3f (rcut=%.3f)\n",
               nx, ny, hx, hy, rcut);

        printf("\n--- Cell bounds ---\n");
        for (int cy = 0; cy < ny; ++cy)
        {
            for (int cx = 0; cx < nx; ++cx)
            {
                int idx = cy * nx + cx;

                double x0 = cx * hx;
                double x1 = (cx + 1) * hx;
                double y0 = cy * hy;
                double y1 = (cy + 1) * hy;

                printf("Cell %2d: cx=%d cy=%d  x:[%.3f, %.3f] y:[%.3f, %.3f]\n",
                       idx, cx, cy, x0, x1, y0, y1);
            }
        }

        cells.resize(nx * ny);
        cell_neighbors.resize(nx * ny);

        printf("\n--- Cell neighbors (9 per cell) ---\n");

        for (int cy = 0; cy < ny; ++cy)
        {
            for (int cx = 0; cx < nx; ++cx)
            {

                int idx = cy * nx + cx;
                int k = 0;

                printf("Cell %2d neighbors:", idx);

                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {

                        int ncx = (cx + dx + nx) % nx;
                        int ncy = (cy + dy + ny) % ny;
                        int nidx = ncy * nx + ncx;

                        cell_neighbors[idx][k++] = nidx;

                        printf(" %d", nidx);
                    }
                }

                printf("\n");
            }
        }
    }

    void clear()
    {
        for (auto &c : cells)
            c.clear();
    }

    void build(const std::vector<Particle> &particles)
    {
        clear();

        for (int i = 0; i < (int)particles.size(); ++i)
        {
            double xp = particles[i].x[0];
            double yp = particles[i].x[1];

            // periodic wrap to [0, L)
            xp = fmod(xp, Lx);
            if (xp < 0)
                xp += Lx;

            yp = fmod(yp, Ly);
            if (yp < 0)
                yp += Ly;

            int cx = int(xp / hx);
            int cy = int(yp / hy);

            cells[cy * nx + cx].push_back(i);
        }
    }

    template <typename Func>
    void find_neighbors(int pid, const std::vector<Particle> &particles, Func callback) const
    {
        const Particle &p = particles[pid];

        // wrap coordinates into [0,L)
        double xp = fmod(p.x[0], Lx);
        if (xp < 0)
            xp += Lx;

        double yp = fmod(p.x[1], Ly);
        if (yp < 0)
            yp += Ly;

        int cx = int(xp / hx);
        int cy = int(yp / hy);

        // search 3×3 neighborhood
        for (int dy_cell = -1; dy_cell <= 1; ++dy_cell)
        {
            for (int dx_cell = -1; dx_cell <= 1; ++dx_cell)
            {
                int ncx = (cx + dx_cell + nx) % nx;
                int ncy = (cy + dy_cell + ny) % ny;

                const auto &cell = cells[ncy * nx + ncx];

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
    }

private:
    double rcut, Lx, Ly;
    int nx, ny;
    double hx, hy;

    std::vector<std::vector<int>> cells;            // cells
    std::vector<std::array<int, 9>> cell_neighbors; // predefined neigbor cells
};
