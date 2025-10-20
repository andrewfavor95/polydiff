# Multi polymer diffusion

Clone this repository using git, then inference can be run when you have downloaded model weights and the necessary apptainer to run (see below).

To download model weights, [click here](https://files.ipd.uw.edu/afavor/train_session2024-07-08_1720455712_BFF_3.00.pt)

This software runs using apptainers. To download the associated apptainer `.sif` file, [click here](https://files.ipd.uw.edu/afavor/SE3nv.sif)

Once you have these files, you can follow the [design tutorial](https://github.com/andrewfavor95/polydiff/blob/main/RFDpoly_tutorial.pdf) to see various design tasks and the associated documentation/syntax.

Contact afavor@uw.edu if you have trouble accessing files or hit any bugs.



# Installation and setup

0. optional: get on an interactive gpu node (example command for slurm):
```Bash
qlogin -p gpu --gres=gpu:a4000:1 -c 2 --mem=16g
```

1. Set relevant paths for installation and navigate to directory for setup (change accordingly given your filesystem and choice of storage location):
```Bash
RFDPOLY_PATH=~/git/RFDpoly_paper_version/

```
2. Navigate to the directory where you want to set up the repo:
```Bash
cd $RFDPOLY_PATH
```

3. clone the repository:
```Bash
git clone git@github.com:andrewfavor95/polydiff.git
```

4. Download model weights and apptainer file:
```Bash
... curl or someshit for weights
... curl or someshit for apptainer sif.
```

5. Set paths for model weights and apptainer .sif file:
```Bash
MODEL_WEIGHTS_PATH=~/public/train_session2024-07-08_1720455712_BFF_3.00.pt
APPTAINER_PATH=~/public/SE3nv.sif
```


3. Run example inference script:
```Bash

$APPTAINER_PATH $RFDPOLY_PATH/polydiff/rf_diffusion/run_inference.py --config-name=multi_polymer \
diffuser.T=50 \
inference.num_designs=3 \
contigmap.contigs=[\'33\ 33\ 75\'] \
contigmap.polymer_chains=[\'dna\',\'rna\',\'protein\'] \
inference.output_prefix=$RFDPOLY_PATH/test_outputs/basic_uncond_test01


```
The initial run will take a little while to precompute the IGSO3 cache, but subsequent runs will be more direct and quick.
If the example command above works, proceed to exploration of the full [design tutorial](https://github.com/andrewfavor95/polydiff/blob/main/RFDpoly_tutorial.pdf).


