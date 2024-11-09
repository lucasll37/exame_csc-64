# README.md

## Project Overview
This project is a parallelized C application designed to optimize runtime by leveraging OpenMP and hash tables for data processing. The goal is to process records from two data files, apply filters, and combine results to generate output efficiently.

### Features
- Parallelized data processing with OpenMP
- Use of hash tables for fast data lookup (via uthash library)
- Dynamic memory allocation
- Comprehensive error handling

### Project Structure
```
project_root/
|-- build/                    # Directory where compiled executables are placed
|   |-- db                    # Executable for generating the database
|   |-- par                   # Parallel executable
|   |-- seq                   # Parallel executable
|-- db/                       # Directory containing input data files
|   |-- A.txt                 # Input data file A
|   |-- B.txt                 # Input data file B
|   |-- ids.txt               # Input IDs file
|-- doc/                      # Documentation files
|   |-- project.md            # Project description
|   |-- solution.md           # Detailed solution documentation
|-- libs/                     # External libraries
|   |-- uthash/               # uthash library for hash table implementation
|-- output/                   # Directory for output CSV files
|   |-- par.csv               # Output file generated by parallel program
|   |-- seq.csv               # Output file generated by sequential program
|   |-- sorted_par.csv        # Sorted output file
|   |-- sorted_seq.csv        # Sorted output file     
|   |-- unique_sorted_seq.cs  # New output with duplicates removed
|-- src/                      # Directory containing source code files
|   |-- db.c                  # Source file for database generation
|   |-- par.c                 # Source file for parallel version
|   |-- seq.c                 # Source file for sequential version
|-- .gitignore                # Git ignore file
|-- LICENSE                   # License file
|-- Makefile                  # Build instructions
|-- README.md                 # Project overview and instructions
|-- TUTORIAL.md               # Step-by-step guide for installation and usage
```

### Requirements
- C Compiler (GCC recommended)
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
1. To generate synthetic data:
   ```bash
   make db
   ```

2. To compiler and run the parallel program:
   ```bash
   make par
   ```

3. To compiler and run the sequential program:
   ```bash
   make seq
   ```
Note: The application is configured to process NUM_RECORDS records. For real-time processing with large datasets (e.g., 30,000 records), ensure your system has sufficient memory and processing power. For testing or demonstration purposes, you can reduce NUM_RECORDS in the source code (e.g., to 50~100) to observe the application's behavior without extensive resource usage.

### Output
The processed output files are saved in the `output/` directory:
`par.csv` and `sorted_par.csv`: Results from the parallel program.
`seq.csv` and `sorted_seq.csv`: Results from the sequential program.
`unique_sorted_seq.csv`: Deduplicated and sorted output.

### Documentation
#### Tutorial
For a detailed, step-by-step guide on how to install the necessary tools, compile, and run the application, please refer to `TUTORIAL.md`. This tutorial includes:

- Installation Instructions: How to install GCC, OpenMP, and any other dependencies.

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