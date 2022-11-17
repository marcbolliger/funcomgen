#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mem=32G
#SBATCH --output=/home/marcbo/dataprep/log/%j.out
#SBATCH --error=/home/marcbo/dataprep/log/%j.err

#Job script to prepare the funcom dataset for use in mlmfc
# Path to store: /itet-stor/marcbo/net_scratch/funcomprep/trainvaltest/
OUTPATH=/itet-stor/marcbo/net_scratch/funcomprep/trainvaltest/
CONDAENV=/itet-stor/marcbo/codesearch-attacks_itetnas04/envs/pythonenv

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
            echo 'Failed to create temp directory' >&2
                exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Activate the conda environment
#source /home/marcbo/.bashrc
[[ -f /itet-stor/marcbo/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/marcbo/net_scratch/conda/bin/conda shell.bash hook)"
conda activate CONDAENV
echo "Conda activated"

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"


# Run the preparation script
python3 dataprep/funcomgen/datasetcreation.py OUTPATH OUTPATH OUTPATH

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
