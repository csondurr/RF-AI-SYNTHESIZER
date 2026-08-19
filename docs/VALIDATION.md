# Validation record

## Current evidence

The model is trained on a synthetic repeated-pole filter family. It has no independent measured-filter dataset, untouched external test set, uncertainty calibration, or production RF-design certification.

## Interpretation rules

- Analytical values are calculations, not measurements.
- Simulated values depend on the model, solver settings, and parameter range.
- Software test results establish behavior only for the tested inputs and environment.
- Hardware claims require calibrated physical measurements and uncertainty reporting.

## Release checklist

1. Record exact software and toolchain versions.
2. Preserve inputs, configuration, random seeds, and generated metrics.
3. Repeat the workflow from a clean checkout.
4. Compare archived and reproduced outputs.
5. Add calibrated physical measurements before making hardware-performance claims.
