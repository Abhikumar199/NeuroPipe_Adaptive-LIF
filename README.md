# NeuroPipe: Adaptive-LIF Spiking Neural Network for Sensor-Based Pipeline Monitoring

NeuroPipe is an experimental neuromorphic-inspired pipeline monitoring project. It explores **time-to-first-spike (TTFS) encoding** and **Adaptive Leaky Integrate-and-Fire (Adaptive-LIF) neurons** for classifying flow and sensor conditions.

The current repository contains a sewer-monitoring dataset derived from the NEXUS testbed, training notebooks, an Adaptive-LIF implementation, and saved evaluation artifacts. A configurable pneumatic leak-monitoring pipeline is also explored in the research notebook; the pneumatic configuration is a proposed extension and should not be interpreted as a validated physical testbed.

## Project overview

The current trained classifier distinguishes four sensor-condition classes:

| Label | Class |
|---:|---|
| 0 | Normal clean-water flow |
| 1 | Contaminated flow |
| 2 | Muddy water / partial blockage |
| 3 | Severe blockage |

The checked-in training script uses three available input features:

- `Flow_L_min`
- `TDS_smooth`
- `Turbidity_smooth`

The pipeline cleans invalid values, performs a stratified train/validation/test split, scales features using the training split, converts normalized values into TTFS input spikes, and trains a feed-forward Adaptive-LIF network using surrogate gradients.

## How the model works

1. **Sensor features:** flow rate, smoothed total dissolved solids (TDS), and smoothed turbidity.
2. **Preprocessing:** remove invalid/infinite rows and scale features to the [0, 1] range using training data only.
3. **TTFS encoding:** each input feature can emit at most one spike in the simulation window. Larger normalized values produce earlier spikes.
4. **Adaptive-LIF dynamics:** membrane potential integrates incoming current with leak. A spike-history-dependent adaptation term raises the effective firing threshold after activity.
5. **Readout:** output spike counts are used as class scores.

The implementation uses a fast-sigmoid surrogate gradient to make the spiking network trainable with backpropagation. This is a software simulation; it is not a deployed neuromorphic chip.

## Repository structure

```text
NeuroPipe_Adaptive-LIF/
├── Training/
│   ├── NeuroPipe.ipynb
│   ├── ann_training.ipynb
│   ├── snn_training.ipynb
│   └── final_data.csv
├── raw_data/
│   ├── final_bloackage.txt
│   ├── muddy+soap_anomly.txt
│   ├── normal_baseline.txt
│   ├── preprocessing.ipynb
│   ├── soap_round_final.txt
│   └── soap_round_light.txt
└── results_neuropipe_adaptive/
    ├── NeuroPipe_adaptive_framework.ipynb
    ├── NeuroPipe_random_stratified_optimization.ipynb
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── metrics.json
    ├── training_history.csv
    └── ttfs_adaptive_lif.pt
```

## Current evaluation

The committed `metrics.json` reports results on a held-out test set of 709 samples, after cleaning 3,545 rows.

| Metric | Test result |
|---|---:|
| Accuracy | 66.15% |
| Macro precision | 63.88% |
| Macro recall | 64.80% |
| Macro F1-score | 63.89% |
| Weighted F1-score | 65.68% |
| Mean spikes per sample | 23.14 |

### Per-class results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Normal clean-water flow | 0.81 | 0.95 | 0.88 | 194 |
| Contaminated flow | 0.69 | 0.61 | 0.65 | 219 |
| Muddy water / partial blockage | 0.60 | 0.49 | 0.54 | 173 |
| Severe blockage | 0.45 | 0.54 | 0.49 | 123 |

These are the results recorded in the repository artifacts, not a claim of field performance. In particular, severe-blockage classification remains a weakness and needs further investigation.

![Adaptive-LIF confusion matrix](results_neuropipe_adaptive/confusion_matrix.png)

## Installation

Python 3.10+ is recommended. Create and activate a virtual environment, then install the core dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib torch jupyter
```

For the extended adaptive-framework notebook, additional dependencies may be required:

```bash
pip install scipy PyWavelets
```

Install a PyTorch build appropriate for your CPU or CUDA setup from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Training

The `Training/NeuroPipe.ipynb` notebook contains the TTFS Adaptive-LIF training implementation. Its script-style entry point accepts the following arguments:

| Argument | Default | Description |
|---|---:|---|
| `--data` | `Training/final_data.csv` | Input CSV |
| `--outdir` | `results_neuropipe_adaptive` | Output directory |
| `--epochs` | 100 | Maximum training epochs |
| `--batch-size` | 64 | Mini-batch size |
| `--hidden` | 32 | Hidden-layer width |
| `--timesteps` | 20 | Simulation time steps |
| `--lr` | 0.001 | Learning rate |

If you extract the script from the notebook into a Python file named `train_neuropipe.py`, run:

```bash
python train_neuropipe.py --data Training/final_data.csv --outdir results_neuropipe_adaptive
```

The training routine uses AdamW, class-weighted cross-entropy, gradient clipping, and early stopping based on validation loss. It saves the model checkpoint, training history, metrics, classification report, and confusion matrix to the output directory.

Alternatively, open the notebook and run its cells in order after checking the configured paths.

## Extended framework notebook

`results_neuropipe_adaptive/NeuroPipe_adaptive_framework.ipynb` explores a more general sensor-processing pipeline. It documents:

- Configurable sensor profiles for the NEXUS sewer dataset and a proposed pneumatic leak-monitoring setup.
- Detrending, optional band-pass filtering, and stationary-wavelet denoising.
- Multi-level features: time-domain, STFT band-energy, and sensor-level statistics.
- TTFS, population-TTFS, and rate-coded input options.
- Adaptive-LIF network experiments and evaluation utilities.

The pneumatic profile describes a target configuration for future acoustic, pressure, vibration, temperature, and flow sensing. Hardware integration and real-world validation are separate future work.

## Outputs

- `ttfs_adaptive_lif.pt`: trained model parameters and preprocessing metadata.
- `metrics.json`: aggregate test metrics and experiment configuration.
- `classification_report.txt`: per-class precision, recall, and F1.
- `confusion_matrix.png`: test-set confusion matrix.
- `training_history.csv`: epoch-wise training and validation history.

## Limitations and next steps

- The current checked-in classifier uses three features from the available NEXUS-derived CSV.
- Results are based on the dataset and split recorded in this repository; they do not establish generalization to unseen sites, pipe geometries, or sensor hardware.
- Severe-blockage recognition needs improvement.
- Chronological and recording-level validation should be prioritized to assess robustness beyond random stratified splits.
- The pneumatic leak-monitoring configuration requires data collection, sensor integration, and physical testbed validation.
- Embedded deployment, latency, power, and energy measurements require separate hardware-level experiments.

## Citation

If you use this repository, please cite or acknowledge the project repository:

```text
Kumar, Abhishek. NeuroPipe: Adaptive-LIF Spiking Neural Network for Sensor-Based Pipeline Monitoring.
GitHub: https://github.com/Abhikumar199/NeuroPipe_Adaptive-LIF
```

## Author

**Abhishek Kumar**  
GitHub: [@Abhikumar199](https://github.com/Abhikumar199)

---

This repository is an experimental research implementation. Reported metrics describe the committed offline evaluation artifacts and should be interpreted in that context.
