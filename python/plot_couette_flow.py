#!/usr/bin/env python3
"""
Plot velocity profile from Couette flow VTU snapshots.
Compares simulation results with analytical solution.
"""

import vtk
import numpy as np
import argparse
import sys
import pathlib
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt

def load_vtu(filename='example.vtu', dx=0.0):
    """Load particle velocity data from VTU file."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    
    output = reader.GetOutput()
    point_data = output.GetPointData()
    
    p_array = point_data.GetArray("p")
    type_array = point_data.GetArray("type")
    v_array = point_data.GetArray("v")
    
    if p_array is None or type_array is None or v_array is None:
        raise RuntimeError("VTU missing required fields: p, type, v")
    
    p = np.array([p_array.GetValue(i) for i in range(p_array.GetNumberOfTuples())])
    type_id = np.array([type_array.GetValue(i) for i in range(type_array.GetNumberOfTuples())])
    
    n_points = v_array.GetNumberOfTuples()
    u = np.zeros(n_points)
    v = np.zeros(n_points)
    for i in range(n_points):
        val = v_array.GetTuple3(i)
        u[i] = val[0]
        v[i] = val[1]
    
    points = output.GetPoints()
    pos_array = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
    x = pos_array[:, 0]
    y = pos_array[:, 1]

    data_points_y = []
    data_vels_x = []
    for i in range(n_points):
        data_points_y.append(y[i])
        data_vels_x.append(u[i])

    return data_points_y, data_vels_x

def main():
    parser = argparse.ArgumentParser(description='Plot Couette flow velocity profile')
    parser.add_argument('vtu_dir', nargs='?', default='.', help='Directory with VTU files')
    parser.add_argument('--out', default='couette_flow.png', help='Output filename')
    args = parser.parse_args()

    # Enable LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })
    plt.rcParams.update({'font.size': 10})

    # Physical parameters (must match run_couette_flow.py)
    fx = 0.0                    # [m/s**2]
    vwall = 0.002               # [m/s]
    lref = 0.1                  # [m]
    rho0 = 1000.0               # [kg/m**3]
    viscosity = 1.0             # [Pa s]
    dx = lref / 40

    refvel = vwall/2
    print('refvel: ', refvel)
    charvel = refvel
    charL = lref
    RE = (rho0 * charvel * charL) / viscosity
    print('RE: ', RE)

    # Output figure
    fig_width_cm = 8.0
    fig_height_cm = fig_width_cm / 1.4
    fig_width_inches = fig_width_cm / 2.54
    fig_height_inches = fig_height_cm / 2.54

    fig, ax = plt.subplots(1, 1, figsize=(fig_width_inches, fig_height_inches))

    # Load VTU files
    vtu_dir = pathlib.Path(args.vtu_dir)
    filenames = sorted(vtu_dir.glob('couette_flow_00*.vtu'))
    
    if len(filenames) == 0:
        print(f"No VTU files found in {vtu_dir}")
        sys.exit(1)

    # Define timesteps (matches VTU filenames)
    t_steps = [5000, 500, 200, 100]

    dt = 0.002257813
    t_times = [t * dt for t in t_steps]

    # VTU directory
    vtu_dir = pathlib.Path(args.vtu_dir)

    # Select files based on t_steps
    sel_filenames = []
    for t in t_steps:
        filename = vtu_dir / f"couette_flow_{t:05d}.vtu"
        if filename.exists():
            sel_filenames.append(filename)
        else:
            print(f"Warning: file {filename} not found")

    colors = ['red', 'green', 'blue', 'orange']
    labels = [fr'$t = {t:.2f}$s' for t in t_times]

    plt.grid(True, zorder=0)

    # Analytical solution: centered coordinate system (0 to +lref)
    xaxis = np.linspace(0, lref, num=200)
    vel_ana_dict = {}

    for t_idx, t_val in enumerate(t_times):
        vel_ana = []
        for i in range(xaxis.shape[0]):
            # Couette profile: v(x) = (vwall)/(lref) * x
            # Poiseuille profile: v(x) = (rho*fx)/(2*mu) * x * (x - lref)
            # with x in [-lref/2, lref/2]
            # vel_temp = (rho0 * fx) / (2.0 * viscosity) * (xaxis[i] + lref/2) * (xaxis[i] - lref/2)
            vel_temp = (vwall/lref)*xaxis[i]

            # Add transient Fourier series
            series = np.linspace(1, 100, num=100)
            vel_temp_sum = 0.0
            for n in range(len(series)):
                vel_temp_sum += ((2*vwall)/(series[n]*np.pi))*(-1)**series[n]*np.sin((series[n]*np.pi*xaxis[i])/lref) \
                    * np.exp(-(viscosity*series[n]**2*np.pi**2*t_val)/(rho0*lref**2))
            # for n in range(len(series)):
            #     vel_temp_sum += (4 * fx * rho0 * lref**2) / (viscosity * np.pi**3 * (2*series[n]+1)**3) \
            #         * np.sin(((np.pi * (xaxis[i] + lref/2)) / lref) * (2*series[n]+1)) \
            #         * np.exp(-(((2*series[n]+1)**2 * np.pi**2 * viscosity) / (rho0 * lref**2)) * t_val)
            
            vel_temp = vel_temp + vel_temp_sum
            vel_ana.append(vel_temp)
        
        vel_ana = np.asarray(vel_ana)
        # vel_ana = -vel_ana  # Negate for flow direction
        vel_ana_dict[t_idx] = vel_ana

        ax.plot(vel_ana, xaxis-lref/2.0, color=colors[t_idx], linewidth=0.8, zorder=5)

    # Plot simulation results
    for i, filename in enumerate(sel_filenames):
        if i < len(labels):
            points_y_sim, vel_x_sim = load_vtu(str(filename), dx)
            points_y_sim = np.asarray(points_y_sim)
            points_y_sim = points_y_sim
            ax.scatter(vel_x_sim, points_y_sim, s=7.0, color=colors[i], label=labels[i], zorder=5)

    ax.set_ylim([-lref/2, lref/2])
    ax.set_xlim([0.0, 0.002])

    y_ticks = np.linspace(-lref/2, lref/2, 5)
    x_ticks = np.linspace(0.0, 0.002, 5)
    ax.set_yticks(y_ticks)
    ax.set_xticks(x_ticks)

    ax.set_ylabel(r'$\mathrm{x}_2 \,[\mathrm{m}]$', fontsize=10)
    ax.set_xlabel(r'$\mathrm{v}_{\mathrm{1}}\,[\mathrm{m}/\mathrm{s}]$', fontsize=10)

    formatterx = ScalarFormatter(useMathText=True)
    formatterx.set_powerlimits((2, 3))
    formatterx.set_scientific(True)
    formattery = ScalarFormatter(useMathText=True)
    formattery.set_powerlimits((0, 0))
    formattery.set_scientific(True)
    ax.yaxis.set_major_formatter(formattery)
    ax.xaxis.set_major_formatter(formatterx)

    ax.legend(fontsize=6, fancybox=False)

    # plt.show()
    # plt.savefig(args.out, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(args.out, bbox_inches='tight', pad_inches=0.02)

    print(f"Saved plot to {args.out}")


if __name__ == '__main__':
    main()
