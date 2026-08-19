# RF AI Synthesizer

A MATLAB deep-learning project for estimating the **center frequency** and **bandwidth** of a simulated RF bandpass filter from its noisy magnitude and unwrapped phase responses.

The project generates a synthetic dual-channel RF dataset, trains a convolutional regression network, saves the trained model, and provides a bilingual English/Turkish desktop interface for interactive inference and visualization.

> **Türkçe özet:** Bu proje, gürültülü genlik ve faz cevaplarından bir RF band-geçiren filtrenin merkez frekansını ve bant genişliğini tahmin eden MATLAB tabanlı bir 1D-CNN sistemidir. Eğitim, model kaydetme, etkileşimli simülasyon ve İngilizce/Türkçe arayüz içerir.

## Table of contents

- [Project objective](#project-objective)
- [How the system works](#how-the-system-works)
- [Signal and filter model](#signal-and-filter-model)
- [Dataset generation](#dataset-generation)
- [Neural-network architecture](#neural-network-architecture)
- [Training configuration](#training-configuration)
- [Interactive application](#interactive-application)
- [Requirements](#requirements)
- [Installation and usage](#installation-and-usage)
- [Project files](#project-files)
- [Inputs and outputs](#inputs-and-outputs)
- [Results and interpretation](#results-and-interpretation)
- [Important limitations](#important-limitations)
- [Reproducibility and validation](#reproducibility-and-validation)
- [Troubleshooting](#troubleshooting)
- [Possible extensions](#possible-extensions)
- [References](#references)

## Project objective

Conventional extraction of RF-filter parameters from measured S-parameter curves may require manual inspection, thresholding, or curve fitting. This project demonstrates a data-driven alternative: a CNN learns the inverse mapping

```text
noisy magnitude response + noisy phase response
                         ↓
             center frequency + bandwidth
```

The current model predicts two continuous parameters:

- Center frequency, `fc`, in the range **4.5–8.5 GHz**
- Bandwidth scaling parameter, `BW`, in the range **0.1–2.0 GHz**

The filter order is fixed at **N = 5** and is not predicted.

## How the system works

1. Random filter parameters are sampled from the configured ranges.
2. A synthetic complex frequency response is calculated at 1001 points between 2 and 11 GHz.
3. Gaussian noise is added independently to magnitude and phase.
4. Magnitude and phase are stored as two input channels.
5. The target center frequency and bandwidth are normalized to `[0, 1]`.
6. A CNN is trained as a two-output regression model.
7. The trained network is saved as `Kusursuz_AI_Model.mat`.
8. The GUI generates a new noisy response, runs inference, denormalizes the prediction, plots both channels, and displays percentage errors.

## Signal and filter model

### Frequency axis

The response is sampled at **1001 linearly spaced points**:

```matlab
freq = linspace(2e9, 11e9, 1001);
```

The broad 2–11 GHz observation window provides context on both sides of every supported center frequency.

### Repeated-pole low-pass prototype

The report uses the following fifth-order prototype:

```text
H_lp(s) = 1 / (1 + s)^N,    N = 5
```

With `s = jΩ`:

```text
H_lp(jΩ) = 1 / (1 + jΩ)^N
```

This is a simple repeated-pole prototype. It is **not the standard Butterworth pole distribution**.

### Frequency transformation

For frequency `f` in GHz, center frequency `fc`, and bandwidth parameter `BW`:

```text
Ω = (f - fc) / BW
H(f) = 1 / (1 + jΩ)^5
```

The clean magnitude and phase are therefore:

```text
Magnitude_dB(f) = 20 log10(|H(f)|)
Phase(f)        = unwrap(angle(H(f)))
```

The MATLAB implementation adds `1e-6` before the logarithm to prevent numerical problems near zero:

```matlab
mag_db = 20 * log10(abs(complex_response) + 1e-6);
```

### Meaning of “bandwidth”

In this model, `BW` is the scale factor in `Ω = (f-fc)/BW`. It should not be interpreted as the conventional full 3 dB bandwidth. At `|f-fc| = BW`, the fifth-order repeated-pole response is approximately 15 dB below its peak. Consequently, the learned output is the model's bandwidth parameter, not automatically a measured VNA 3 dB bandwidth.

## Dataset generation

The training script creates **2500 synthetic samples**.

| Property | Value |
|---|---:|
| Number of samples | 2500 |
| Frequency points per sample | 1001 |
| Frequency range | 2–11 GHz |
| Input channels | 2 |
| Channel 1 | Magnitude in dB |
| Channel 2 | Unwrapped phase in radians |
| Center-frequency range | 4.5–8.5 GHz |
| Bandwidth-parameter range | 0.1–2.0 GHz |
| Fixed filter order | 5 |
| Holdout ratio | 20% |
| Approximate training samples | 2000 |
| Approximate validation/test samples | 500 |

### Parameter sampling

Both targets are sampled uniformly:

```matlab
f_c_ghz = 4.5 + rand(1) * 4.0;
b_w_ghz = 0.1 + rand(1) * 1.9;
```

### Noise model

The training data uses additive Gaussian noise:

| Channel | Training noise |
|---|---:|
| Magnitude | `0.3 * randn` dB |
| Phase | `0.02 * randn` rad |

The application exposes an environmental-noise slider from 0 to 1. In the GUI, its value scales the same maximum standard deviations:

```matlab
magnitude noise = noiseLvl * 0.3
phase noise     = noiseLvl * 0.02
```

### Tensor layout

CNN inputs are stored in a four-dimensional MATLAB array:

```text
[frequency points, singleton width, channels, samples]
[1001,             1,               2,        2500]
```

In code:

```matlab
X_data = zeros(1001, 1, 2, 2500);
```

The two regression targets are stored as `[fc_norm, bw_norm]`.

### Target normalization

```text
fc_norm = (fc - 4.5) / 4.0
bw_norm = (BW - 0.1) / 1.9
```

The application reverses this transformation after inference:

```text
fc = fc_norm × 4.0 + 4.5
BW = bw_norm × 1.9 + 0.1
```

## Neural-network architecture

Although MATLAB represents the input with `imageInputLayer` and `convolution2dLayer`, all convolution and pooling kernels have width 1. The network therefore operates along the frequency axis like a **1D CNN with two input channels**.

| Stage | Layer | Configuration |
|---|---|---|
| Input | Image input | `[1001 1 2]`, z-score normalization |
| Block 1 | Convolution | 32 filters, kernel `[21 1]`, same padding |
|  | Batch normalization | — |
|  | Leaky ReLU | slope 0.01 |
|  | Max pooling | pool `[2 1]`, stride `[2 1]` |
| Block 2 | Convolution | 64 filters, kernel `[11 1]`, same padding |
|  | Batch normalization | — |
|  | Leaky ReLU | slope 0.01 |
|  | Max pooling | pool `[2 1]`, stride `[2 1]` |
| Block 3 | Convolution | 128 filters, kernel `[5 1]`, same padding |
|  | Batch normalization | — |
|  | Leaky ReLU | slope 0.01 |
|  | Max pooling | pool `[2 1]`, stride `[2 1]` |
| Regression head | Fully connected | 128 units |
|  | Leaky ReLU | slope 0.01 |
|  | Dropout | probability 0.1 |
| Output | Fully connected | 2 outputs |
| Loss | Regression layer | regression/MSE-style objective |

Large kernels in the first block capture broad spectral structure. Later, smaller kernels and deeper feature maps refine local magnitude and phase patterns. Pooling progressively compresses the frequency dimension.

## Training configuration

The network is trained with Adam:

| Option | Value |
|---|---:|
| Optimizer | Adam |
| Maximum epochs | 200 |
| Mini-batch size | 64 |
| Initial learning rate | 0.001 |
| Learning-rate schedule | Piecewise |
| Drop period | 60 epochs |
| Drop factor | 0.5 |
| Validation data | 20% holdout set |
| Training-progress window | Enabled |
| Command-window verbosity | Disabled |

The learning rate is halved every 60 epochs. After training, the complete network object is written to:

```text
Kusursuz_AI_Model.mat
```

Depending on MATLAB release and hardware, training time can vary substantially.

## Interactive application

The desktop application provides two tabs:

- **English Interface**
- **Türkçe Arayüz**

Both tabs use the same trained network and expose:

- Center-frequency spinner: 4.5–8.5 GHz
- Bandwidth spinner: 0.1–2.0 GHz
- Environmental-noise slider: 0–1
- Inference/simulation button
- Magnitude-response plot
- Unwrapped-phase plot
- Predicted center frequency
- Predicted bandwidth parameter
- Percentage error for each prediction

The default input values are:

```text
fc = 6.5 GHz
BW = 0.75 GHz
noise factor = 0.3
```

Percentage errors are calculated as:

```text
fc error (%) = |fc_true - fc_pred| / fc_true × 100
BW error (%) = |BW_true - BW_pred| / BW_true × 100
```

Errors below 5% are shown in green; larger errors are shown in yellow.

## Requirements

- MATLAB
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox, for `cvpartition`
- A MATLAB release supporting `uifigure`, `uitabgroup`, `uiaxes`, `uispinner`, and `uislider`
- Sufficient memory for the `1001 × 1 × 2 × 2500` input tensor and CNN training
- Optional: a supported GPU and Parallel Computing Toolbox for accelerated training

No external dataset is required because the training data is generated synthetically.

## Installation and usage

### 1. Download the repository

```bash
git clone https://github.com/csondurr/RF-AI-SYNTHESIZER.git
cd RF-AI-SYNTHESIZER
```

You may also download the repository as a ZIP from GitHub.

### 2. Open the project folder in MATLAB

Make the repository folder the current MATLAB working directory so that the saved model and application are resolved correctly.

### 3. Train the model

Run:

```matlab
main
```

This generates the synthetic dataset, performs the holdout split, trains the network, opens MATLAB's training-progress plot, and saves `Kusursuz_AI_Model.mat` in the current folder.

### 4. Start the application

Run the application directly:

```matlab
Uygulama
```

The primary function now matches the repository file name, so no manual rename is required.

The application will stop with an error dialog if `Kusursuz_AI_Model.mat` is not available in the current MATLAB path.

## Project files

| File | Purpose |
|---|---|
| `main.m` | Generates data, defines the CNN, trains it, validates during training, and saves the model |
| `Uygulama.m` | Defines the bilingual interactive inference application |
| `RF Filter AI Synthesizer.pdf` | Technical report covering theory, equations, architecture, training, discussion, and references |
| `Kusursuz_AI_Model.mat` | Generated trained model; created after running `main.m` |
| `README.md` | Project documentation |

> The trained `.mat` model can be large. If it is not committed to the repository, users must run `main.m` before launching the application.

## Inputs and outputs

### Training input

Each synthetic example contains:

- A 1001-point noisy magnitude curve in dB
- A 1001-point noisy unwrapped-phase curve in radians

### Training target

Each example contains two normalized scalar targets:

- Normalized center frequency
- Normalized bandwidth parameter

### Application output

The GUI displays denormalized physical-unit predictions:

- Predicted `fc` in GHz
- Predicted `BW` in GHz
- Relative percentage error for each parameter
- The generated magnitude and phase curves

The final fully connected layer is unconstrained. Therefore, under unusual or out-of-distribution inputs, normalized predictions may fall outside `[0, 1]`, producing physical predictions outside the nominal training ranges.

## Results and interpretation

The accompanying report describes a typical decrease in training and validation loss and cites **expected**, not repository-benchmarked, normalized RMSE values around 0.01–0.02. These would correspond approximately to:

- Center-frequency absolute error: about 40–80 MHz over a 4 GHz range
- Bandwidth-parameter absolute error: about 19–38 MHz over a 1.9 GHz range

These figures should be treated as expectations until a fixed-seed evaluation script publishes measured metrics from a saved model. The current source does not calculate or save final RMSE, MAE, R², confidence intervals, or per-sample prediction tables.

## Important limitations

- **Synthetic-only training:** the network is not trained or validated on real VNA/S-parameter measurements.
- **Fixed response family:** all examples follow `1/(1+jΩ)^5`.
- **Fixed order:** `N = 5`; the network does not estimate filter order.
- **Bandwidth definition:** `BW` is a model scale parameter, not conventional full 3 dB bandwidth.
- **Validation/test reuse:** the 20% holdout is supplied as validation data during training; there is no separate untouched test set.
- **No fixed random seed:** dataset generation, noise, and partitioning vary from run to run.
- **No persisted dataset:** generated samples exist in memory and are not saved.
- **No final metric script:** performance claims are not automatically reproduced by the current code.
- **No output constraint:** predictions are not clipped to the training ranges.
- **Domain shift risk:** cable loss, calibration errors, impedance mismatch, ripple, resonances, nonlinear frequency grids, and real instrument noise are not modeled.
- **Naming mismatch:** `Uygulama.m` and the primary function `AI_Bilingual_App` should be made consistent.
- **Memory allocation:** arrays use MATLAB's default double precision; `single` precision could reduce memory usage.

This project is a research and educational prototype, not a calibrated replacement for RF measurement or production test equipment.

## Reproducibility and validation

For reproducible experiments, place a fixed seed near the start of `main.m`:

```matlab
rng(42, "twister");
```

A stronger experimental protocol would use separate training, validation, and test partitions. After training, evaluate the untouched test set and report at least:

- MAE and RMSE in normalized units
- MAE and RMSE in GHz or MHz
- R² for center frequency and bandwidth
- Worst-case and percentile errors
- Error versus noise level
- Error versus center frequency and bandwidth
- Multiple runs with different seeds

The GUI's displayed error is calculated against the synthetic parameters used to generate that individual curve. It is not a confidence score or generalization metric.

## Troubleshooting

### `Kusursuz_AI_Model.mat not found`

Run `main.m` first and verify that MATLAB saved `Kusursuz_AI_Model.mat` in the current folder or somewhere on the MATLAB path.

### Undefined function `AI_Bilingual_App`

Rename `Uygulama.m` to `AI_Bilingual_App.m`, or rename the primary function to `Uygulama` as described in [Installation and usage](#installation-and-usage).

### Undefined function or class related to deep learning

Install and license MATLAB Deep Learning Toolbox.

### `cvpartition` is unavailable

Install and license Statistics and Machine Learning Toolbox.

### Out-of-memory during training

Try one or more of the following:

- Reduce `num_samples`
- Reduce `MiniBatchSize`
- Store arrays as `single`
- Use a machine with more RAM
- Use a supported GPU configuration

### Results change between runs

This is expected because parameter sampling, noise generation, weight initialization, and the holdout split are random. Add a fixed `rng` seed when repeatability is required.

### GUI opens but prediction fails

Confirm that the loaded `.mat` file contains a variable named exactly `net` and that it was trained with the same 1001-point, two-channel input format.

## Possible extensions

- Train and fine-tune on measured Touchstone/VNA data
- Support S11, S21, group delay, or additional channels
- Predict filter order, insertion loss, return loss, ripple, Q factor, and topology
- Generate Butterworth, Chebyshev, elliptic, and Bessel response families
- Model calibration errors, impedance mismatch, cable loss, drift, and colored/non-Gaussian noise
- Use a separate validation set and untouched test set
- Add automated RMSE, MAE, R², plots, and exported result tables
- Add prediction intervals or uncertainty estimation
- Constrain or clip outputs to physically meaningful ranges
- Package the interface with MATLAB App Designer
- Export the model for deployment or real-time measurement workflows

## References

1. MathWorks, *MATLAB Deep Learning Toolbox Documentation*.
2. A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, Prentice Hall, 2010.
3. I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*, MIT Press, 2016.
4. D. M. Pozar, *Microwave Engineering*, Wiley, 2011.

## Disclaimer

This repository is intended for education and experimentation. Predictions are based on a simplified simulated filter family and must be independently verified before use in RF design, calibration, quality-control, or safety-critical workflows.


## Repository maintenance

**Evidence boundary:** Synthetic-model validation only. Generated GUI predictions must not be interpreted as measured filter parameters or production-ready synthesis results.

- [Validation record](docs/VALIDATION.md)
- [Contribution guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Citation metadata](CITATION.cff)

## License

Copyright (c) 2026 Cem Sondur. Distributed under the [MIT License](LICENSE). Third-party components remain subject to their original licenses.
