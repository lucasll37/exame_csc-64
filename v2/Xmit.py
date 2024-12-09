import os
import sys

if len(sys.argv) < 3:
    raise ValueError("Usage: Xmit.py <ranks MPI> <threads OpenMP>")

RanksMPI = int(sys.argv[1])
ThreadsOMP = int(sys.argv[2])

# Calculate the number of nodes required
NODES = RanksMPI // 24
REM = RanksMPI % 24
if REM > 0:
    NODES += 1

# Define job parameters
pwd = os.getcwd()
Exec = f"{pwd}/build/par"
Job = f"V_{RanksMPI}_{ThreadsOMP}"
ArqSaida = f"{pwd}/OutTempo_{RanksMPI}MPI_{ThreadsOMP}OMP.out"

# Generate the ssub.sh script
ssub_content = f"""#!/bin/bash
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node=24
#SBATCH --ntasks={RanksMPI}
#SBATCH --cpus-per-task={ThreadsOMP}
#SBATCH -p sequana_cpu_dev
#SBATCH -J {Job}
#SBATCH --time=00:05:00
#SBATCH --output={ArqSaida}
#SBATCH --exclusive

cd $SLURM_SUBMIT_DIR
ulimit -s unlimited
export OMP_NUM_THREADS={ThreadsOMP}
srun -n $SLURM_NTASKS --mpi=openmpi -c $SLURM_CPUS_PER_TASK {Exec}
"""

with open("ssub.sh", "w") as fOut:
    fOut.write(ssub_content)

# Make the script executable and submit it
os.system(f"chmod +x {pwd}/ssub.sh")
os.system("sbatch ssub.sh")