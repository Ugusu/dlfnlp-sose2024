#!/bin/bash
#SBATCH --job-name=train-best-sst-result
#SBATCH -t 00:10:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                    # take 1 GPU, see https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html for more options
#SBATCH --mem-per-gpu=8G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=8            # number cores per task
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=TODO@stud.uni-goettingen.de
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

module load anaconda3
source activate dnlp # Or whatever you called your environment.

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env 2> /dev/null

# Print out some git info.
module load git
echo -e "\nCurrent Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Latest Commit: $(git rev-parse --short HEAD)"
echo -e "Uncommitted Changes: $(git status --porcelain | wc -l)\n"

# Run the script with the specified hyperparameters:
python -u multitask_classifier.py \
  --use_gpu \
  --local_files_only \
  --option finetune \
  --task sst \
  --pooling attention \
  --regularize_context \
  --lr 0.00005 \
  --hidden_dropout_prob 0.5 \
  --batch_size 64 \
  --optimizer adamw \
  --epochs 5

# Check live logs (replace <jobid> with your actual job ID)
# tail -f slurm_files/slurm-train-multitask_classifier-<jobid>.out
# tail -f slurm_files/slurm-train-multitask_classifier-<jobid>.err
