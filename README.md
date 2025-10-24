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
mkdir -p RFDPOLY_DIR
mkdir -p WEIGHTS_DIR
mkdir -p ENV_DIR
mkdir -p DESIGN_DIR

```

2. Navigate to the directory where you want to set up the repo, and clone it:
```Bash
cd $RFDPOLY_PATH
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

```

4. Navigate to the directory where you want to apptainer file, download it, and set path variable:
```Bash
cd $ENV_DIR
curl -O https://files.ipd.uw.edu/pub/2025_RFDpoly/SE3nv.sif
APPTAINER_PATH=$ENV_DIR/SE3nv.sif

```

## Testing that everything works:
Change directory to wherever you want to perform design, and run example inference script:
```Bash
cd $DESIGN_DIR

$APPTAINER_PATH $RFDPOLY_PATH/polydiff/rf_diffusion/run_inference.py --config-name=multi_polymer \
inference.ckpt_path=$MODEL_WEIGHTS_PATH \
diffuser.T=50 \
inference.num_designs=3 \
contigmap.contigs=[\'33\ 33\ 75\'] \
contigmap.polymer_chains=[\'dna\',\'rna\',\'protein\'] \
inference.output_prefix=$RFDPOLY_PATH/test_outputs/basic_uncond_test01

```
The initial run will take a little while to precompute the IGSO3 cache, but subsequent runs will be more direct and quick.


If the example command above works, proceed to exploration of the full [design tutorial](https://github.com/andrewfavor95/polydiff/blob/main/RFDpoly_tutorial.pdf).

