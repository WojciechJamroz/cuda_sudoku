<div align="center">

# 🚀 CUDA-Accelerated Sudoku Solver 🚀

[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CuPy](https://img.shields.io/badge/CuPy-10.0+-FF6F00?style=for-the-badge&logo=numpy&logoColor=white)](https://cupy.dev/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA_RTX-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com/)

**A high-performance Sudoku solving system leveraging the massive parallelism of NVIDIA GPUs via CUDA. Capable of evaluating hundreds of millions of potential Sudoku board solutions per second.**

[Features](#-key-features) •
[Requirements](#-requirements) •
[Installation](#%EF%B8%8F-installation) •
[Usage](#-usage) •
[How It Works](#-how-it-works) •
[Performance](#-performance-results) •
[Advanced Tips](#-advanced-optimization-tips)

</div>

---

## 📋 Table of Contents
- [🌟 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [💻 Requirements](#-requirements)
- [🛠️ Installation](#%EF%B8%8F-installation)
- [📂 Project Structure](#-project-structure)
- [🚀 Usage](#-usage)
- [⚙️ Command Line Arguments](#%EF%B8%8F-command-line-arguments)
- [💡 How It Works](#-how-it-works)
- [📊 Performance Results](#-performance-results)
- [✅ Recommended Configuration](#-recommended-configuration)
- [⚠️ Limitations](#%EF%B8%8F-limitations)
- [🚀 Advanced Optimization Tips](#-advanced-optimization-tips)
- [🔧 Troubleshooting](#-troubleshooting)
- [📜 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

---

## 🚦 Quick Start
```bash
# Install dependencies (for CUDA 12.x)
pip install cupy-cuda12x numpy

# Run with default settings (Quick Test, Smart Solver, Easy Puzzle)
python main_cuda.py

# Try solving a hard puzzle with advanced solver
python main_cuda.py --solver advanced --puzzle-difficulty hard --batch-size 16777216
```

---

## 🌟 Overview

This project tackles the classic Sudoku puzzle using a non-traditional, massively parallel approach. Instead of relying solely on deterministic algorithms like backtracking or constraint propagation, it harnesses the power of CUDA to explore the vast solution space through **intelligent, constraint-guided randomization**.

By generating and validating millions of candidate boards simultaneously on the GPU, this solver can rapidly find solutions, especially for puzzles with numerous possibilities or those where traditional methods might get stuck. It's a demonstration of GPU computing applied to combinatorial search problems.

<details>
<summary><b>💡 Why use a GPU for Sudoku?</b></summary>

While traditional CPU-based algorithms like backtracking can solve many Sudoku puzzles efficiently, they explore solutions sequentially. The GPU approach:

1. Explores millions of potential solutions in parallel
2. Can escape "difficult valleys" in the solution space through massive exploration
3. Demonstrates an alternative approach to combinatorial optimization
4. Showcases CUDA programming techniques like shared memory, bit manipulation, and kernel optimization

</details>

---

## ✨ Key Features

*   **⚡ Ultra-High Performance**: Achieves peak speeds exceeding **200-300 million board checks per second** on modern NVIDIA GPUs (e.g., RTX 4080 SUPER).
*   **🧠 Multiple Solving Algorithms**:
    *   **Original Solver**: A baseline random generation approach. Fills empty cells purely randomly and checks validity. Simple but less efficient for harder puzzles.
    *   **Smart Solver**: Significantly faster. Fills empty cells *only* with numbers that are valid according to row, column, and box constraints at that moment. Reduces the search space dramatically.
    *   **Advanced Solver**: Builds upon the Smart Solver. Introduces a **difficulty bias** heuristic, prioritizing potentially harder-to-fill cells or exploring alternative valid numbers probabilistically. Adaptively switches strategies for challenging puzzles.
*   **⚙️ Advanced Optimization Techniques**:
    *   **Shared Memory Utilization**: Kernels use GPU shared memory for faster data access compared to global memory, caching board data locally within thread blocks.
    *   **Early Termination**: Validity checks within the CUDA kernel stop immediately upon finding the first rule violation (duplicate number in a row, column, or box), saving computation.
    *   **Bit Manipulation**: Uses bitmasks (`unsigned short`) for extremely fast checking of seen numbers within rows, columns, and boxes.
    *   **Adaptive Batch Sizing**: (Optional) Dynamically adjusts the number of boards processed per iteration based on puzzle difficulty and observed performance.
    *   **Optimized Kernel Launch Configuration**: Tuned block and grid sizes for efficient GPU utilization.
    *   **Asynchronous Operations**: Leverages CUDA streams for potentially overlapping data transfers and kernel execution (though the current structure is largely synchronous per batch).
*   **📊 Comprehensive Benchmarking**: Includes a built-in testing framework (`--mode 2`) to evaluate solver performance across different puzzle difficulties and configurations.

---

## 💻 Requirements

*   **Python**: Version 3.7 or higher.
*   **NVIDIA GPU**: A CUDA-capable GPU is essential. Compute Capability 6.0+ recommended.
*   **CUDA Toolkit**: While CuPy bundles runtime libraries, having the full Toolkit installed (matching the driver version) can be beneficial for development or troubleshooting. Version 11.x or 12.x recommended.
*   **NVIDIA Driver**: A recent driver compatible with your CUDA Toolkit version.
*   **CuPy**: Version 10.0 or higher. Install the version matching your CUDA Toolkit (e.g., `cupy-cuda11x`, `cupy-cuda12x`).
*   **NumPy**: Required for initial puzzle setup and some utility functions.

---

## 🛠️ Installation

1.  **Ensure CUDA and NVIDIA Drivers are installed correctly.** You can verify with `nvidia-smi` in your terminal.
2.  **Install Python dependencies:** Choose the CuPy package matching your installed CUDA Toolkit version (e.g., 11.x or 12.x).

    ```bash
    # Example for CUDA 12.x
    pip install cupy-cuda12x numpy

    # Example for CUDA 11.x
    # pip install cupy-cuda11x numpy
    ```

<details>
<summary><b>🔄 Verifying your installation</b></summary>

Run this quick check to verify your CUDA setup:

```python
import cupy as cp
print(f"CUDA available: {cp.cuda.is_available()}")
print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
print(f"Device count: {cp.cuda.runtime.getDeviceCount()}")
print(f"Current device: {cp.cuda.Device().attributes}")
```

</details>

---

## 📂 Project Structure

```
cuda_test/
├── main_cuda.py        # Main script: handles arguments, orchestrates solving/benchmarking
├── kernels.cu          # Contains the raw CUDA C++ kernel code (loaded by CuPy)
├── puzzles.py          # Stores the example Sudoku puzzles
├── utils.py            # Utility functions (e.g., printing boards)
└── README.md           # This file
```

---

## 🚀 Usage

Execute the main script from your terminal.

```bash
# Basic usage (Quick Test, Smart Solver, Easy Puzzle, Default Batch Size)
python main_cuda.py

# Quick test a specific medium puzzle using the Smart Solver
python main_cuda.py --mode 1 --solver smart --puzzle-difficulty medium --puzzle-index 1

# Run the Advanced Solver on a hard puzzle with a large batch size and higher bias
python main_cuda.py --mode 1 --solver advanced --batch-size 16777216 --difficulty-bias 5.0 --puzzle-difficulty hard

# Run a full benchmark across easy, medium, and hard puzzles using the Smart Solver
# Note: Benchmarking can take several minutes.
python main_cuda.py --mode 2 --solver smart --batch-size 8388608 --max-attempts 500000000

# Run benchmark using all solvers (will take longer)
python main_cuda.py --mode 2 --solver all --batch-size 4194304 --max-attempts 100000000 # Adjust batch/attempts as needed
```

---

## ⚙️ Command Line Arguments

| Argument            | Type    | Description                                                                 | Default     | Example Usage                     |
| :------------------ | :------ | :-------------------------------------------------------------------------- | :---------- | :-------------------------------- |
| `--mode`            | `int`   | **1**: Quick Test (single puzzle), **2**: Full Benchmark (multiple puzzles) | `1`         | `--mode 2`                        |
| `--solver`          | `str`   | Algorithm: `original`, `smart`, `advanced`, `all` (for benchmark)           | `smart`     | `--solver advanced`               |
| `--batch-size`      | `int`   | Number of boards processed in parallel per GPU kernel launch.               | `4,194,304` | `--batch-size 8388608`            |
| `--max-attempts`    | `int`   | Stop if no solution is found after this many total attempts.                | `100,000,000` | `--max-attempts 500000000`        |
| `--adaptive`        | `flag`  | Enable adaptive batch sizing for the Smart Solver.                          | `False`     | `--adaptive`                      |
| `--difficulty-bias` | `float` | Bias factor (1.0-10.0) for the Advanced Solver. Higher values add more randomness for difficult cells. | `3.0`       | `--difficulty-bias 6.5`           |
| `--puzzle-difficulty` | `str`   | Puzzle difficulty for Quick Test (`mode 1`): `easy`, `medium`, `hard`       | `easy`      | `--puzzle-difficulty hard`        |
| `--puzzle-index`    | `int`   | Index of the puzzle within the chosen difficulty level (`mode 1`).          | `0`         | `--puzzle-index 1`                |

---

## 💡 How It Works

The core idea is parallelized guess-and-check, optimized for the GPU.

1.  **Initialization**:
    *   The initial Sudoku puzzle (a 9x9 grid) is loaded.
    *   Empty cells (represented by `0`) are identified.
    *   The puzzle state and empty cell coordinates are transferred to GPU memory (using CuPy `cp.array`).
    *   Index arrays (`board_indices`, `empty_rows`, `empty_cols`) are pre-calculated on the GPU to efficiently map random numbers to the correct cells across the entire batch.

2.  **Batch Processing Loop**:
    *   **Template Creation**: A batch of identical copies of the initial puzzle board is created on the GPU (`batch_boards_gpu`). The size of this batch is determined by `batch_size`.
    *   **Parallel Filling**: A CUDA kernel (`smart_random_fill` or `advanced_smart_fill`) is launched. Thousands or millions of GPU threads execute in parallel:
        *   Each thread is responsible for filling *one specific empty cell* in *one specific board* within the batch.
        *   **(Smart/Advanced)**: The kernel calculates valid number candidates for that cell based on current row, column, and 3x3 box constraints *within that board*.
        *   A pseudo-random number generator (seeded) selects one of the valid candidates (or a purely random number if no valid options exist or if using the Original/random fallback).
        *   **(Advanced)**: The `difficulty_bias` influences the selection probability, potentially exploring less obvious choices for cells with few options.
    *   **Parallel Validation**: A second CUDA kernel (`check_sudoku_batch`) is launched.
        *   Each thread is responsible for validating *one entire board* from the batch.
        *   It checks rows, columns, and 3x3 boxes for duplicates using efficient bitmasking and early termination.
        *   The result (valid or invalid) for each board is written to a results array (`results_gpu`).
    *   **Result Aggregation**: The CPU checks the `results_gpu` array (e.g., using `cp.any()`). This implicitly synchronizes the GPU work for that batch.
    *   **Success Check**: If any board in the batch is valid, its index is found (`cp.argmax()`), the board is copied back from GPU to CPU memory (`.get()`), printed, and the program terminates successfully.
    *   **Iteration**: If no solution is found, the loop repeats with a new random seed for the filling kernel.

3.  **Termination**: The loop continues until a solution is found, the maximum attempt limit (`max_attempts`) is reached, or the user interrupts (Ctrl+C).

<details>
<summary><b>🔍 CUDA Kernels Deep Dive</b></summary>

*   `check_sudoku_batch`: Highly optimized for speed. Uses `__restrict__` pointers, shared memory (`shared_board`) for data locality within a block, `#pragma unroll` for loop optimization, and bitwise operations (`seen_mask`) for O(1) duplicate checking per number. Early termination (`&& is_valid`) is crucial.
*   `smart_random_fill`: Focuses on constraint calculation. Also uses bitmasks (`invalid_mask`) to quickly determine disallowed numbers for a cell. Generates a list of valid `options` and randomly selects one.
*   `advanced_smart_fill`: Extends the smart fill. Incorporates the `cell_difficulty_bias` to potentially select an alternative valid number based on a probability calculation, adding heuristic guidance.

</details>

---

## 📊 Performance Results

Performance heavily depends on your specific GPU, CPU, puzzle difficulty, and chosen parameters. The following are **representative results** obtained on an **NVIDIA RTX 4080 SUPER**:

| Solver     | Typical Speed (attempts/sec) | Optimization Level | Notes                                                                 |
| :--------- | :--------------------------- | :----------------- | :-------------------------------------------------------------------- |
| Original   | ~50-70 Million               | Basic              | Pure random fill, less effective but simple.                          |
| **Smart**  | **~200-330+ Million**        | High               | Constraint-based fill, significantly faster for most puzzles.         |
| Advanced   | ~2-10 Million                | Maximum            | More complex kernel, lower raw speed but potentially more *effective* for very hard puzzles due to heuristic exploration. Speed varies greatly with bias and batch size. |

*   **Why the speed difference?** The Smart Solver prunes the search space effectively by only placing valid numbers. The Advanced Solver adds heuristic overhead per attempt but might navigate tricky solution paths more effectively by exploring less probable (but still valid) numbers. Raw speed isn't the only metric; time-to-solution for difficult puzzles matters.
*   Larger batch sizes generally increase throughput (attempts/sec) up to the point where the GPU is fully utilized or limited by memory bandwidth/kernel complexity.

---

## ✅ Recommended Configuration

Tailor settings for best results based on puzzle difficulty (number of empty cells and their constraints):

*   **Easy Puzzles (~15-30 empty cells)**:
    *   `--solver smart`
    *   `--batch-size 8388608` (8M) to `16777216` (16M)
    *   `--adaptive` can be useful.
*   **Medium Puzzles (~30-45 empty cells)**:
    *   `--solver smart` or `--solver advanced`
    *   `--batch-size 16777216` (16M) to `33554432` (32M)
    *   If using Advanced: `--difficulty-bias 3.0` to `5.0`
*   **Hard Puzzles (45+ empty cells, especially constrained ones)**:
    *   `--solver advanced` is often necessary.
    *   `--batch-size 33554432` (32M) to `67108864` (64M) or more, memory permitting.
    *   `--difficulty-bias 5.0` to `7.0` (higher bias encourages more exploration).
    *   Increase `--max-attempts` significantly (e.g., `500000000` or higher).

---

## ⚠️ Limitations

*   **Not Guaranteed**: This is a probabilistic solver. It's *not guaranteed* to find a solution, especially for extremely difficult puzzles within the attempt limit.
*   **No Uniqueness Check**: It finds *a* solution. If multiple solutions exist, it returns the first one found by chance.
*   **Memory Intensive**: Large batch sizes consume significant GPU VRAM. Monitor usage if you push batch sizes very high.
*   **Overhead**: While fast, there's overhead in data transfer (minimal here) and kernel launch compared to a purely CPU-based deterministic solver *for very simple puzzles*. The benefit shines with parallelism needs.

---

## 🚀 Advanced Optimization Tips

*   **Tune Batch Size**: This is the most critical parameter. Experiment to find the sweet spot for your GPU and the puzzle difficulty. Too small = underutilized GPU; too large = memory issues or diminishing returns.
*   **Adjust Difficulty Bias**: For the `advanced` solver on hard puzzles, carefully tuning the bias (e.g., `--difficulty-bias 6.0`) can sometimes unlock solutions faster by balancing constraint adherence with exploration.
*   **Increase Max Attempts**: For notoriously hard Sudokus, you might need billions of attempts (`--max-attempts 2000000000`).
*   **Enable Adaptive Sizing**: Use `--adaptive` with the `smart` solver to let the script try to adjust the batch size automatically based on observed performance (experimental).
*   **Profile**: Use NVIDIA Nsight Systems or Nsight Compute to profile kernel performance and identify bottlenecks if you plan further optimization.

---

## 🔧 Troubleshooting

*   **CuPy Installation Issues**:
    *   Ensure you have the correct NVIDIA driver and CUDA Toolkit version installed *before* installing CuPy.
    *   Install the specific CuPy package matching your CUDA Toolkit (e.g., `pip install cupy-cuda12x` for CUDA 12.x).
    *   Consult the official [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html).
*   **`cupy.cuda.compiler.CompileException`**: This often means the CUDA Toolkit's compiler (`nvcc`) cannot be found or there's an issue compiling the kernels in `kernels.cu`. Ensure the CUDA Toolkit's `bin` directory is in your system's PATH environment variable. Verify `nvcc --version` works in your terminal.
*   **Slow Performance**:
    *   Ensure you are using a dedicated NVIDIA GPU and not integrated graphics. Check `nvidia-smi`.
    *   Experiment with `--batch-size`. The default might not be optimal for your specific GPU.
    *   Make sure background processes aren't heavily utilizing the GPU.
*   **Out of Memory Errors**: Reduce the `--batch-size`. Very large batch sizes consume significant VRAM.

<details>
<summary><b>📋 Common Error Solutions</b></summary>

**Error: CuPy cannot be imported**
```
ModuleNotFoundError: No module named 'cupy'
```
**Solution**: Install the correct CuPy version. Example: `pip install cupy-cuda12x`

**Error: CUDA out of memory**
```
cudaErrorMemoryAllocation: out of memory
```
**Solution**: Reduce batch size using `--batch-size` parameter. Try halving it from the default.

**Error: Kernel does not compile**
```
cupy.cuda.compiler.CompileException: ... nvcc ...
```
**Solution**: Check CUDA installation; make sure `nvcc` is in your PATH and `nvidia-smi` works.

</details>

---

## 💼 Contributing

Contributions to improve the solver are welcome! Here are some ideas:

- Implement additional solving algorithms
- Optimize existing CUDA kernels for better performance
- Add visualization tools for solution progress
- Support for larger or custom Sudoku variants (e.g., 16x16)

Please feel free to submit pull requests or open issues for discussion.

---

## 📜 License

This project is licensed under the **MIT License**. See the LICENSE file for details.

---

## 🙏 Acknowledgements

This project serves as an example of applying GPU acceleration to combinatorial problems. It highlights how massive parallelism, combined with optimized CUDA kernels and smart heuristics, can tackle complex search spaces efficiently.

<div align="right">
<a href="#-cuda-accelerated-sudoku-solver-">⬆️ Back to top</a>
</div>