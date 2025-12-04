#include "vtk_writer.hpp"

void VTKWriter::write(const std::vector<Particle> &particles, int step)
{
    // Standard filename
    std::ostringstream filename;
    filename << "particles_" << std::setw(5) << std::setfill('0') << step << ".vtu";
    write(particles, step, filename.str());
}

void VTKWriter::write(const std::vector<Particle> &particles, int /*step*/, const std::string &filename)
{
    std::ofstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "ERROR: cannot write " << filename << std::endl;
        return;
    }

    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    f << "  <UnstructuredGrid>\n";
    f << "    <Piece NumberOfPoints=\"" << particles.size()
      << "\" NumberOfCells=\"" << particles.size() << "\">\n";

    // --- POINT DATA ---
    f << "      <PointData Scalars=\"p rho type\" Vectors=\"v vf\">\n";
    f << "        <DataArray type=\"Float64\" Name=\"p\" format=\"ascii\">\n";
    for (auto &p : particles)
        f << " " << p.p;
    f << "\n        </DataArray>\n";

    f << "        <DataArray type=\"Float64\" Name=\"rho\" format=\"ascii\">\n";
    for (auto &p : particles)
        f << " " << p.rho;
    f << "\n        </DataArray>\n";

    f << "        <DataArray type=\"Int32\" Name=\"type\" format=\"ascii\">\n";
    for (auto &p : particles)
        f << " " << p.type;
    f << "\n        </DataArray>\n";

    f << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"v\" format=\"ascii\">\n";
    for (auto &p : particles)
        f << " " << p.v[0] << " " << p.v[1] << " 0.0";
    f << "\n        </DataArray>\n";

    f << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"vf\" format=\"ascii\">\n";
    for (auto &p : particles)
    {
        if (p.vf.has_value())
            f << " " << (*p.vf)[0] << " " << (*p.vf)[1] << " 0.0";
        else
            f << " 0.0 0.0 0.0";
    }
    f << "\n        </DataArray>\n";

    f << "        <DataArray type=\"Float64\" Name=\"pf\" format=\"ascii\">\n";
    for (auto &p : particles)
    {
        if (p.pf.has_value())
            f << " " << *p.pf;
        else
            f << " 0.0";
    }
    f << "\n        </DataArray>\n";
    f << "      </PointData>\n";

    // --- POINTS ---
    f << "      <Points>\n";
    f << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (auto &p : particles)
        f << " " << p.x[0] << " " << p.x[1] << " 0.0";
    f << "\n        </DataArray>\n";
    f << "      </Points>\n";

    // --- CELLS ---
    f << "      <Cells>\n";
    f << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < particles.size(); ++i)
        f << " " << i;
    f << "\n        </DataArray>\n";

    f << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 1; i <= particles.size(); ++i)
        f << " " << i;
    f << "\n        </DataArray>\n";

    f << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < particles.size(); ++i)
        f << " 1";
    f << "\n        </DataArray>\n";

    f << "      </Cells>\n";
    f << "    </Piece>\n";
    f << "  </UnstructuredGrid>\n";
    f << "</VTKFile>\n";

    f.close();
}