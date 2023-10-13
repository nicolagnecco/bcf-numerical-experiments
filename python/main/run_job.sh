#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=pr_12345 -A pr_12345
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N test
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e job_${PBS_JOBID}.err
#PBS -o job_${PBS_JOBID}.log
### Only send mail when job is aborted or terminates abnormally
#PBS -m n
### Number of nodes
#PBS -l nodes=1:ppn=8 
### Memory
#PBS -l mem=120gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 12 hours)
#PBS -l walltime=12:00:00
 
# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

### Here follows the user commands:
# Define number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS nodes

# Load all required modules (specific to your cluster) for the job
module load tools
module load perl/5.20.2
module load miniconda3/4.10.3

# Create a conda environment and install requirements
conda create --prefix ~/envs -y
conda activate ~/envs
pip install pip --upgrade
pip install -r requirements.txt
pip install -e .


# This is where the work is done
# Make sure that this script is not bigger than 64kb ~ 150 lines, 
# otherwise put in separate script and execute from here
$1

conda deactivate
