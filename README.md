# Multi polymer diffusion

## Basic info:
This repository contains the code for the [associated preprint](https://www.biorxiv.org/content/10.1101/2025.10.01.679929v1), which can be cited
```bib
@article{favor2025novo,
  title={De novo design of RNA and nucleoprotein complexes},
  author={Favor, Andrew H and Quijano, Riley and Chernova, Elizaveta and Kubaney, Andrew and Weidle, Connor and Esler, Morgan A and McHugh, Lilian and Carr, Ann and Hsia, Yang and Juergens, David and others},
  journal={bioRxiv},
  pages={2025--10},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

You can clone this repository using git, then inference can be run when you have downloaded model weights and the necessary apptainer to run (see below).

To download model weights:
  * [click here for the best RNA-design weights](https://files.ipd.uw.edu/pub/2025_RFDpoly/train_session2024-07-08_1720455712_BFF_3.00.pt)
  * [click here for the best general multi-polymer weights](https://files.ipd.uw.edu/pub/2025_RFDpoly/train_session2024-06-27_1719522052_BFF_7.00.pt)

This software runs using apptainers. To download the associated apptainer `.sif` file, [click here](https://files.ipd.uw.edu/pub/2025_RFDpoly/SE3nv.sif)
Equivalent setup instructions for conda are in progress.

Once you have these files, you can follow the [design tutorial](https://github.com/andrewfavor95/polydiff/blob/main/RFDpoly_tutorial.pdf) to see various design tasks and the associated documentation/syntax.

Contact afavor@uw.edu if you have trouble accessing files or hit any bugs.





## Installation and setup

1. Set relevant paths for installation and navigate to directory for setup (change accordingly given your filesystem and choice of storage location):
```Bash
# Where you choose to install things (CHANGE THESE):
RFDPOLY_DIR=~/git/RFDpoly_paper_version
WEIGHTS_DIR=~/git/RFDpoly_paper_version/weights
ENV_DIR=~/git/RFDpoly_paper_version/exec

# Wherever you want to run some design jobs (ALSO CHANGE THIS):
DESIGN_DIR=~/git/RFDpoly_paper_version/design_jobs

# Make the directories
mkdir -p $RFDPOLY_DIR
mkdir -p $WEIGHTS_DIR
mkdir -p $ENV_DIR
mkdir -p $DESIGN_DIR

```

2. Navigate to the directory where you want to set up the repo, and clone it:
```Bash
cd $RFDPOLY_DIR
git clone git@github.com:andrewfavor95/polydiff.git

```

3. Navigate to the directory where you want to save model weights, download them, and set path variable for weights of your choice:
```Bash
# Download model weights:
cd $WEIGHTS_DIR

# Best weights for RNA-only design:
curl -O https://files.ipd.uw.edu/pub/2025_RFDpoly/train_session2024-06-27_1719522052_BFF_7.00.pt
# Best weights for generalized design across all polymer classes:
curl -O https://files.ipd.uw.edu/pub/2025_RFDpoly/train_session2024-07-08_1720455712_BFF_3.00.pt

# Set weights path (replace with your choice of weights):
MODEL_WEIGHTS_PATH=$WEIGHTS_DIR/train_session2024-07-08_1720455712_BFF_3.00.pt
export RFDPOLY_CKPT_PATH="$MODEL_WEIGHTS_PATH"

```

4. Setting up the environment:
Navigate to the directory where you want to apptainer file, download it, and set path variable:
```Bash
cd $ENV_DIR
curl -O https://files.ipd.uw.edu/pub/2025_RFDpoly/SE3nv.sif
APPTAINER_PATH=$ENV_DIR/SE3nv.sif

```

Equivalent python environments can be set up using conda, or an environment manager of your choice.
All dependencies and packages are specified in the files found in `rf_diffusion/environment` (see `environment.yml` and `SE3nv.spec`) 

## Testing that everything works (the *DEMO*):
Change directory to run from within the RFDpoly directory useful for filepath searches if you are not providing an input pdb), and run example inference script.
Be sure to specify your desired output directory with `$DESIGN_DIR`.
```Bash
cd $DESIGN_DIR/

$APPTAINER_PATH $RFDPOLY_DIR/polydiff/rf_diffusion/run_inference.py --config-name=multi_polymer \
diffuser.T=50 \
inference.num_designs=1 \
contigmap.contigs=[\'33\ 33\ 75\'] \
contigmap.polymer_chains=[\'dna\',\'rna\',\'protein\'] \
inference.output_prefix=$DESIGN_DIR/test_outputs/basic_uncond_test01

```
The initial run will take a little while to precompute the IGSO3 cache, but subsequent runs will be more direct and quick.


if that throws errors, then try:
```
$APPTAINER_PATH $RFDPOLY_DIR/polydiff/rf_diffusion/run_inference.py --config-name=multi_polymer \
diffuser.T=50 \
inference.input_pdb=$RFDPOLY_DIR/polydiff/rf_diffusion/test_data/DBP035.pdb \
inference.num_designs=3 \
contigmap.contigs=[\'33\ 33\ 75\'] \
contigmap.polymer_chains=[\'dna\',\'rna\',\'protein\'] \
inference.output_prefix=$DESIGN_DIR/test_outputs/basic_uncond_test01


```
explanation: model initialization searches for an input pdb filepath, even if you aren't performing motif scaffolding. Providing a (real) dummy filepath will fix this is the default search paths are unsuccessful.

**Expected output:** a three chain .pdb file, containing a complex of DNA (chain A), RNA (chain B), and protein (chain C).


## If the example (demo) command above works, proceed to exploration of the full [design tutorial](https://github.com/andrewfavor95/polydiff/blob/main/RFDpoly_tutorial.pdf).
The full design tutorial contains many inference commands for the types of designs reported in the RFDpoly paper.
The design tutorial also is intented to provide documentation, and explain arguments in the context of their use-cases.



# Additional Software Policy checklist items:
## System requirements

- **Operating systems**
  - Primary workflow: Linux host with [Apptainer](https://apptainer.org/) ≥ 1.1 (or Singularity equivalent). The bundled container image runs Ubuntu 22.04.2 LTS.
  - Alternative workflow: macOS 13+/Windows 11 via Conda (CPU only unless you install CUDA-capable PyTorch wheels).
- **Software dependencies**
  - Apptainer runtime to execute `SE3nv.sif`.
  - For native/Conda installs: Python 3.10, PyTorch 1.13.1, CUDA 11.7 toolchain (if using an NVIDIA GPU), PyRosetta 2023.09+, DGL 1.0.1, e3nn 0.5.1, hydra-core 1.3.2, and the packages listed in `rf_diffusion/environment/environment.yml`.
  - **NOTE:** All dependencies are specified in `rf_diffusion/environment/`, such that users can set up an equivalent working environment on any operating system.
  - Downloaded assets: model checkpoints (`*.pt`, ~2.3 GB each) and the container image (`SE3nv.sif`, ~8 GB).
- **Tested configurations**
  - Ubuntu 22.04 host with Apptainer 1.1.9, NVIDIA driver ≥ 515, CUDA 11.7 runtime.
  - PyTorch 1.13.1+cu117, Python 3.10.8, PyRosetta 2023.09 inside the container.
- **Hardware**
  - Minimum: 16 GB RAM and 40 GB free disk space for the repository, checkpoints, and cache.
  - Recommended: NVIDIA GPU with ≥ 16 GB VRAM for practical throughput. CPU-only execution is supported but 5–10× slower. No additional specialized hardware is required.

## Times to install provided files and run inference (tested on Linux operating system, Ubuntu 24.04.2):
- Downloading apptainer .sif file:
  - Total: 1 minute, 15 seconds
- Downloading model weight .ckpt file:
  - Total: 7 seconds
- Cloning the repository:
  - Total: 52 seconds
- Demo inference run:
  - first run, including generation of IGSO3 cache: Total: 2 minutes, 26 seconds
  - subsequent runs (per trajectory): 50 seconds.

## Testing:
- This software has only been tested on Linux (Ubuntu 24.04.3 LTS).
