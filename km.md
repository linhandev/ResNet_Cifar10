srun --cpus-per-task=2 --mem=8GB --time=00:10:00  --pty /bin/bash

srun --cpus-per-task=4 --mem=16GB --time=10:00:00 --gres=gpu:mi50:1 --pty /bin/bash

singularity exec --overlay /scratch/lh3317/envs/torch-rocm.ext3:ro /scratch/work/public/singularity/hudson/images/rocm4.5.2-ubuntu20.04.3.sif /bin/bash

source /ext3/env.sh
