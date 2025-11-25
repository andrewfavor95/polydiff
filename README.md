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

## Installation and setup

0. Clone the repository 
```bash
git clone git@github.com:RosettaCommons/RFDpoly.git
```

1. Set environment variables and create directories 

Set relevant paths for installation and then create the directories the paths point to:

```bash
# Feel free to change the exact names of the directories
RFDPOLY_DIR=/path/to/RFDpoly
WEIGHTS_DIR=/path/to/RFDpoly/weights

# path for where to put the Apptainer .sif file
ENV_DIR=/path/to/RFDpoly/exec

# Make the directories
mkdir -p $RFDPOLY_DIR
mkdir -p $WEIGHTS_DIR
mkdir -p $ENV_DIR
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
export RFDPOLY_CKPT_PATH=\$WEIGHTS_DIR/train_session2024-07-08_1720455712_BFF_3.00.pt
```

4. Setting up the environment:
Navigate to the directory where you want to [Apptainer](https://apptainer.org/) file, download it, and set path variable:
```Bash
cd $ENV_DIR
curl -O https://files.ipd.uw.edu/pub/2025_RFDpoly/SE3nv.sif
APPTAINER_PATH=$ENV_DIR/SE3nv.sif
```
Downloading the `.sif` file may take several minutes. 

<!--Equivalent python environments can be set up using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), or an environment manager of your choice. Environment files are provided in `rf_diffusion/environment` and can be used as follows:
```bash
apptainer build SE3nv.sif SE
```-->

## Basic Use Case Example: Testing Your Setup

In this example, RFDpoly will create a single design
with three chains:
1. A DNA chain with 33 segments
2. An RNA chain with 33 segments
3. A protein chain with 75 residues

Make a directory to store the outputs of the demo and store 
it in the `DESIGN_DIR` environment variable:
```bash
export DESIGN_DIR=/path/to/your/output/directory
```

```Bash
cd $DESIGN_DIR/

apptainer run --nv $APPTAINER_PATH $RFDPOLY_DIR/rf_diffusion/run_inference.py --config-name=multi_polymer \
diffuser.T=50 \
inference.ckpt_path=$RFDPOLY_CKPT_PATH \
inference.num_designs=1 \
contigmap.contigs=[\'33\ 33\ 75\'] \
contigmap.polymer_chains=[\'dna\',\'rna\',\'protein\'] \
inference.output_prefix=$DESIGN_DIR/test_outputs/basic_uncond_test01
```
The initial run will take a little while to precompute the IGSO3 cache, but subsequent runs will be more direct and quick.


if that throws errors, then try:
```bash
cd $DESIGN_DIR

apptainer run --nv $APPTAINER_PATH $RFDPOLY_DIR/rf_diffusion/run_inference.py --config-name=multi_polymer \
diffuser.T=50 \
inference.ckpt_path=$RFDPOLY_CKPT_PATH \
inference.input_pdb=$RFDPOLY_DIR/rf_diffusion/test_data/DBP035.pdb \
inference.num_designs=1 \
contigmap.contigs=[\'33\ 33\ 75\'] \
contigmap.polymer_chains=[\'dna\',\'rna\',\'protein\'] \
inference.output_prefix=$DESIGN_DIR/test_outputs/basic_uncond_test01
```

explanation: model initialization searches for an input pdb filepath, even if you aren't performing motif scaffolding. Providing a (real) dummy filepath will fix this is the default search paths are unsuccessful.

**NOTE:** using the `--config-name=multi_polymer` specification is the best way to ensure that all settings work together as expected, and are consistent with the behavior reported in the manuscript.

## If the example (demo) command above works, proceed to exploration of the full [design tutorial](https://github.com/RosettaCommons/RFDpoly/blob/main/RFDpoly_tutorial.pdf).
The full design tutorial contains many inference commands for the types of designs reported in the RFDpoly paper.
The design tutorial is intended to provide documentation, and explain arguments in the context of their use-cases.

Contact **afavor@uw.edu** if you have trouble accessing files or hit any bugs.

# Additional Software Policy checklist items:
## System requirements

- **Operating systems**
  - Primary workflow: Linux host with [Apptainer](https://apptainer.org/) ≥ 1.1 (or Singularity equivalent). The bundled container image runs Ubuntu 22.04.2 LTS.
  - Alternative workflow: macOS 13+/Windows 11 via Conda (CPU only unless you install CUDA-capable PyTorch wheels).
- **Software dependencies**
  - Apptainer runtime to execute `SE3nv.sif`, this can be built from `SE3nv.spec` in `rf_diffusion/environment`
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
