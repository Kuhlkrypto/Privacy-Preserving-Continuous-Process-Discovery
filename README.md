# Privacy-Preserving Continuous Process Discovery

This repository contains the implementation, evaluation framework, and research report for a novel continuous process discovery pipeline that provides user-level Differential Privacy (DP). 

The goal of this research project is to enable the discovery of Directly-Follows Graphs (DFGs) from infinite event data streams without exhausting the privacy budget, while maintaining a bounded memory footprint and structural model utility.

## Project Structure

The repository is organized into the following key components:


  - `src/`: Core logic including private (`private_discovery`) and non-private (`non_private_discovery`) DFG generation, stream handling, and metrics.
  - `run_experiments.py`: Main script to execute experiments across different hyperparameter configurations (e.g., window size, publication frequency, trace truncation).
  - `test_framework.py`: Testing utilities for the pipeline.
  - `requirements.txt`: Python dependencies for running the simulator.
  - `Dockerfile`: Setup for containerized execution.

- **`Report/`**: LaTeX source code for the final research report.
  - Contains chapters covering Introduction, Background, Methodology, Implementation, and Evaluation.
  - Requires the `tudscrreprt` LaTeX class (TU Dresden) for compilation.


## Getting Started

### Prerequisites

To run the simulator and visualization scripts, you will need Python installed along with the required packages:

```bash
cd Privacy-Preserving-Continuous-Process-Discovery
pip install -r requirements.txt
# Alternatively, you can use uv: uv pip install -r requirements.txt
```

### Running Experiments

To run the DP streaming pipeline evaluations on your event logs:
```bash
cd Privacy-Preserving-Continuous-Process-Discovery
python run_experiments.py
```

### Generating Visualizations

After generating result JSON files (e.g., in `Sepsis/`), you can visualize the results:

```bash
python visualize_results.py Sepsis/W10pct_r5_LQ100.json --save-dir ./plots
```


## Key Features

- **Case Frequency Counting**: Replaces Event Frequency to strictly bound a single user's impact on any DFG edge counter.
- **Dynamic Budget Reclamation**: Allows privacy budgets to travel through temporal sliding windows and be reclaimed once data exits the active window, avoiding budget depletion over infinite streams.
- **Post-Processing Graph Restoration**: Utilizes start activity filtering and network flow conservation optimization to recover graph utility from zero-mean Laplace noise.
