#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=subhamoy@ualberta.ca
#SBATCH --account=def-tati
set -e
module --force purge
module load nixpkgs/16.09  intel/2017.1  openmpi/2.0.2
module load gromacs/5.1.4
module load python/3.7.0
module load scipy-stack
module load networkx

# Import constants
ndna=`python -c 'import constants;print(constants.ndna)'`
npei=`python -c 'import constants;print(constants.npei)'`

#Create Index file for individual molecules
python gen_index.py

#Create a directory to store mindist files
mkdir -p mindist
cpus=$(nproc)
echo "Total available CPUs = $cpus"
for (( d=0;d<${ndna};d++ ))
do
	for (( p=0; p<${npei}; p++))
  	do
			PID=$((ndna+p))
                        run_jobs=`jobs -pr | wc -l`
                        echo "Total running jobs = $run_jobs"
                        while true
                        do 
                              if [ $run_jobs -lt $cpus ]
                              then
                                   break
                              fi
                              sleep 1
                        done
			echo "$d $PID"| gmx mindist -f md_1${i}.xtc -s md_1.tpr -n index_mol.ndx -pbc -od mindist/mindist${d}-${p}.xvg &> mindist.log &
                        echo "DNA $d - PEI $p done"        
  	done
done


