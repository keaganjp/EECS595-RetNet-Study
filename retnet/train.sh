#!/bin/bash -l

#SBATCH --job-name=SDCPROJ
#SBATCH --mail-user=uniqname@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=100g
#SBATCH --time=8:00:00
#SBATCH --account=na595s001f23_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --output=/FILE DIRECTORY/output.log
module purge
module load python3.10-anaconda

cd /home/keaganjp/RetNet
mkdir output_logs/
mkdir output
mkdir checkpoints/
nvidia-smi
python train.py --epochs 10 --batch-size 16 --experiment-name 'trial'