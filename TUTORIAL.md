
# TUTORIAL.md

## Step-by-step Guide to Compile and Run the Project

### Prerequisites
Ensure the following tools are installed on your system:
- GCC (with OpenMP support)
- uthash library

### Setting Up the Environment
1. **Install GCC** (if not already installed):
   ```bash
   sudo apt-get install gcc
   ```
2. **Verify OpenMP Support**:
   ```bash
   gcc -fopenmp --version
   ```
### Compiling the Project
The `Makefile` provided automates the build process and organizes the output. Here’s what each target does:

- **`make all`**: Compiles all necessary executables (parallel, sequential, and database generation).
- **`make db`**: Compiles and runs the database generation executable.
- **`make par`**: Compiles only the parallel version of the program.
- **`make seq`**: Compiles only the sequential version of the program.
- **`make clean`**: Cleans up all generated build and output files.

Run the following to build the entire project:
```bash
make all
```
This command will create the `parallel`, `seq`, and `db` executables in the `build` directory.

### Running the Program
To generate the database files:
```bash
make db
```

To run the parallelized program:
```bash
make par
```

To run the sequential program:
```bash
make seq
```
Obs.: To perform fact-time processing, simulate for NUM_RECORDS values ​​up to 50.

### Understanding the Output
The program processes the input data and creates a CSV file `output_par.csv` with the following columns:
- `ID_a_m`: ID from record A
- `ID_b_M`: ID from record B
- `ID'`: Combined ID
- `a_m`: Value from record A
- `b_M`: Value from record B
- `f`: Final product