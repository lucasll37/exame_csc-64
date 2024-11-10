import os
import sys

nNodes=1
nTasksPerNode=1
nTasks=1

if len(sys.argv) < 2:
    raise ValueError("faltou argumento numero de threads")

nThreads=sys.argv[1]
queue="sequana_cpu_dev"
execTime="00:01:00"

pwd = os.getcwd()
execName = pwd + "/build/par"

job="C_" + nThreads
outFile = pwd + "/Out_Parallel_" + nThreads + ".txt"

fOut=open("ssub.sh", "w")

strOut = f"""#!/bin/bash\n
#SBATCH --nodes={nNodes}
#SBATCH --ntasks-per-node={nTasksPerNode}
#SBATCH --ntasks={nTasks}
#SBATCH --cpus-per-task={nThreads}
#SBATCH -p {queue}
#SBATCH -J {job}
#SBATCH --time={execTime}
#SBATCH --output={outFile}
#SBATCH --exclusive

cd $SLURM_SUBMIT_DIR
ulimit -s unlimited
export OMP_NUM_THREADS={nThreads}
srun -n $SLURM_NTASKS {execName}
exit
"""

fOut.write(strOut)
fOut.close()

os.system(f"chmod +x {pwd}/ssub.sh")
os.system("sbatch ssub.sh")
