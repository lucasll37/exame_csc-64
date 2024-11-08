
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
3. **Install uthash**:
   - Download uthash from [uthash GitHub](https://github.com/troydhanson/uthash) and place the header file `uthash.h` in `./libs/uthash/`.

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
To run the parallelized program:
```bash
./build/parallel
```

To run the sequential program:
```bash
./build/seq
```

To generate the database files:
```bash
./build/db
```

### Understanding the Output
The program processes the input data and creates a CSV file `output_par.csv` with the following columns:
- `ID_a_m`: ID from record A
- `ID_b_M`: ID from record B
- `ID'`: Combined ID
- `a_m`: Value from record A
- `b_M`: Value from record B
- `f`: Final product

### Troubleshooting
- **Error opening file**: Ensure the input files `A.txt`, `B.txt`, and `ids.txt` are present in the `db` directory.
- **Memory allocation error**: Verify sufficient system memory is available.
- **Compilation errors**: Ensure that `uthash.h` is correctly placed in `./libs/uthash/` and included via `INCLUDES` in the `Makefile`.

### Customizing the Build
To modify the compiler or flags, edit the `Makefile`:
```makefile
CC = gcc
CFLAGS = -Wall -fopenmp
INCLUDES = -I./libs/uthash/include
```
This ensures compatibility and optimizes performance with OpenMP and hash table functionalities.

### Additional Notes
- The runtime of the program is displayed upon completion.
- The final sorted output is available as `sorted_output_par.csv` in the `output` directory.

### Contact
For any issues, contact [your_email@domain.com].

