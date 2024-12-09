# CSC-64 Final Exam - Parallel Data Processing

## Project Overview

This project is a parallelized C application designed to optimize runtime by leveraging OpenMP, MPI, and CUDA for data processing. The goal is to process records from two data files, apply filters, and combine results to generate output efficiently.

### Features

- Parallelized data processing with OpenMP, MPI, and CUDA
- Use of hash tables for fast data lookup (via uthash library)
- Dynamic memory allocation
- Comprehensive error handling

### Project Structure

``` plaintext
project_root/
|-- doc/                      # Documentation files
|-- results/                  # Directory for output results
|   |-- v1/                   # Results for version 1
|   |-- v2/                   # Results for version 2
|   |-- v3/                   # Results for version 3
|-- v1/                       # Version 1 source and documentation
|   |-- libs/                 # External libraries for version 1
|   |-- src/                  # Source files for version 1
|   |-- TUTORIAL.md           # Step-by-step guide for installation and execution of the version
|   |-- Makefile              # Build instructions for the version
|-- v2/                       # Version 2 source and documentation
|   |-- libs/                 # External libraries for version 2
|   |-- src/                  # Source files for version 2
|   |-- TUTORIAL.md           # Step-by-step guide for installation and execution of the version
|   |-- Makefile              # Build instructions for the version
|-- v3/                       # Version 3 source and documentation
|   |-- libs/                 # External libraries for version 3
|   |-- src/                  # Source files for version 3
|   |-- TUTORIAL.md           # Step-by-step guide for installation and execution of the version
|   |-- Makefile              # Build instructions for the version
|-- .gitignore                # Git ignore file
|-- LICENSE                   # License file
|-- Makefile                  # Build instructions
|-- README.md                 # Project overview and instructions
```

### Requirements

#### Version 1 (v1)

- C Compiler (GCC recommended)
- OpenMP library

#### Version 2 (v2)

- C Compiler (GCC recommended)
- MPI Compiler (mpicc)
- OpenMP library

#### Version 3 (v3)

- C Compiler (GCC recommended)
- MPI Compiler (mpicc)
- CUDA Compiler (nvcc)
- OpenMP library

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/lucasll37/exame_csc-64.git
   ```

2. Navigate to the project directory:

   ```bash
   cd exame_csc-64
   ```

### Usage

For the proper usage, change your directory to the version you want to run (v1, v2, or v3). Then, you should be able to run the following commands.

#### Compiling

1. To generate synthetic data:

   ```bash
   make db
   ```

2. To compile the parallel program:

   ```bash
   make par
   ```

3. To compile the sequential program:

   ```bash
   make seq
   ```

4. To execute all the above commands:

   ```bash
   make all
   ```

Note: The application is configured to process NUM_RECORDS records. For real-time processing with large datasets (e.g., 30,000 records), ensure your system has sufficient memory and processing power. For testing or demonstration purposes, you can reduce NUM_RECORDS in the source code (e.g., to 50~100) to observe the application's behavior without extensive resource usage.

### Output

The processed output files are saved in the `output/` directory:

- `par.csv`, `sorted_par.csv` and `unique_sorted_par.csv`: Results from the parallel program.
- `seq.csv`, `sorted_seq.csv` and `unique_sorted_seq.csv`: Results from the sequential program.

To analyze the correctness and performance of the parallelized application, compare the final output file (`unique_sorted_par.csv`) with the sequential version (`unique_sorted_seq.csv`). Ensure that the parallel version produces the same results as the sequential version but with improved runtime.

### Documentation

#### Tutorial

For a detailed, step-by-step guide on how to install the necessary tools, compile, and run the application, please refer to:

TUTORIAL.md

. This tutorial includes:

- Installation Instructions: How to install GCC, OpenMP, MPI, CUDA, and any other dependencies.
- Compilation Steps: Commands and explanations for compiling the database generator and both versions of the program.
- Execution Guidance: How to run the executables, adjust parameters, and interpret command-line output.
- Troubleshooting Tips: Common issues and how to resolve them.

#### Solution Details

In-depth documentation about the solution, including the algorithms used, optimizations implemented can be found in `doc/solution.md`. This document covers:

- Problem Overview: Detailed description of the problem and requirements.
- Algorithm Explanation: Insights into the data structures and algorithms employed.
- Optimization Strategies: Discussion of parallelization techniques, memory optimizations, and other enhancements.
- Challenges and Solutions: Problems encountered during development and how they were addressed.

### Cleaning Up

To clean up the build and output files:

```bash
make clean
```

### License

[MIT License](LICENSE)
