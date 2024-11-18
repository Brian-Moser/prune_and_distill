#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then

  # put your install commands here:
  # apt update
  # apt install -y [...]
  # apt clean
  # conda install -y [...]
  
  ### BEGIN LDM STUFF ###
  if [ ! -f src/latentdiff/models/ldm/cin256-v2/model.ckpt ]; then
    echo "Downloading pre-trained LDM ..."
    wget -O src/latentdiff/models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
    echo "Pre-trained LDM downloaded ..."
  else
    echo "Pre-trained LDM already exists ... Skipping download ..."
  fi
  pip install src/taming-transformers-master
  pip install omegaconf>=2.0.0 pytorch-lightning==1.6.5 torch-fidelity einops
  pip install git+https://github.com/openai/CLIP.git
  pip install kornia -U
  ### END LDM STUFF ###
  
  pip install -r requirements.txt

  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi

# This runs your wrapped command
# "$@"
