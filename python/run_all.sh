#!/usr/bin/env bash
set -e  # abort on error

# Global settings
export OMP_NUM_THREADS=4

PYTHON=python3
mkdir -p plots

# Channel flow settings (Poiseuille + Couette)
RES_CHANNEL=40
STEPS_CHANNEL=20001
RE_CHANNEL_LIST=(0.1 10.0)
FLOWS=("poiseuille" "couette")

# Lid Driven Cavity (LDC) settings
RES_LDC=50
STEPS_LDC=20001
RE_LDC_LIST=(100.0 1000.0)

# Run channel flow simulations
echo "=== Running channel flow simulations ==="

for FLOW in "${FLOWS[@]}"; do
  for RE in "${RE_CHANNEL_LIST[@]}"; do
    echo "Running ${FLOW}, Re=${RE}, res=${RES_CHANNEL}"

    ${PYTHON} run_${FLOW}_flow.py \
      --num-threads ${OMP_NUM_THREADS} \
      -Re ${RE} \
      --resolution ${RES_CHANNEL} \
      --steps ${STEPS_CHANNEL}
  done
done

# Plot channel flow results
echo "=== Generating channel flow plots ==="

for FLOW in "${FLOWS[@]}"; do
  for RE in "${RE_CHANNEL_LIST[@]}"; do
    ${PYTHON} plot_${FLOW}_flow.py \
      -Re ${RE} \
      -res ${RES_CHANNEL} \
      -out plots/${FLOW}_Re${RE}_res${RES_CHANNEL}.png
  done
done

# Run LDC simulations
echo "=== Running Lid Driven Cavity simulations ==="

for RE in "${RE_LDC_LIST[@]}"; do
  echo "Running LDC, Re=${RE}, res=${RES_LDC}"

  ${PYTHON} run_lid_driven_cavity_tv.py \
    --num-threads ${OMP_NUM_THREADS} \
    -re ${RE} \
    --resolution ${RES_LDC} \
    --steps ${STEPS_LDC}
done

# Plot LDC results
echo "=== Generating LDC plots ==="

for RE in "${RE_LDC_LIST[@]}"; do
  ${PYTHON} plot_ldc_velocity_profile.py \
    --re ${RE} \
    --res ${RES_LDC} \
    --out plots/ldc_velocity_profile_Re${RE}_res${RES_LDC}.png

  ${PYTHON} plot_ldc_velocity_field.py \
    --re ${RE} \
    --res ${RES_LDC} \
    --out plots/ldc_velocity_field_Re${RE}_res${RES_LDC}.png
done

# Build LaTeX report
echo "=== Building PDF report ==="

cd latex
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
cd ..

echo "=== DONE ==="
