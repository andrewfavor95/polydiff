# Installation Guide

## Using the Apptainer Image
If possible, we highly recommend using the provided apptainer image which can be obtained via:
```bash
curl -O https://files.ipd.uw.edu/pub/2025_RFDpoly/SE3nv.sif
```

Here is an example for how to use this image to run RFDpoly: 
```bash
apptainer run --nv /path/to/SE3nv.sif /path/to/RFDpoly/rf_diffusion/run_inference.py --config-name=multi_polymer \
diffuser.T=50 \
inference.ckpt_path=/path/to/train_session2024-07-08_1720455712_BFF_3.00.pt \
inference.num_designs=1 \
contigmap.contigs=[\'33\ 33\ 75\'] \
contigmap.polymer_chains=[\'dna\',\'rna\',\'protein\'] \
inference.output_prefix=/path/to/your/output/directory/basic_uncond_test01
```

If you see an error when attempting to run this related to
a lack of an input file, try adding this option:
```bash
inference.input_pdb=/path/to/RFDpoly/rf_diffusion/test_data/DBP035.pdb
```

## Creating a Conda Environment

### Linux: 
If the apptainer image is not compatible with your system
or you are doing development with RFDpoly, you can create
a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) environment with a provided `environment.yml` file: 

```bash
conda env create -f /path/to/RFDpoly/rf_diffusion/environment/environment.yml
conda activate RFDpoly_env
pip install /path/to/RFDpoly/rf_diffusion/RF2-allatom/rf2aa/SE3Transformer
```
The last command installs a slightly modified version of 
se3-transformer. If installing this package does not work,
you can try installing [`se3-transformer-pytorch`](https://pypi.org/project/se3-transformer-pytorch/) instead,
but we do not guarantee that your results will be exactly
the same. 

You can test your installation by running: 
```bash
python /path/to/RFDpoly/rf_diffusion/run_inference.py \
--config-name=multi_polymer \
diffuser.T=50 \
inference.ckpt_path=/path/to/RFDpoly/weights/train_session2024-07-08_1720455712_BFF_3.00.pt \
inference.num_designs=1 \
contigmap.contigs=[\'33\ 33\ 75\'] \
contigmap.polymer_chains=[\'dna\',\'rna\',\'protein\'] \
inference.output_prefix=./basic_uncond_test01
```

If using the `environment.yml` file does not work, please
create an issue. Or, if you modified the `.yml` file to 
work for your system, make a pull request to add it to 
the `rf_diffusion/environment` folder. 

### MacOS
If the apptainer image is not compatible with your system
or you are doing development with RFDpoly, you can create
a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) environment with a provided `macos_environment.yml` file: 

```bash
conda env create -f /path/to/RFDpoly/rf_diffusion/environment/macos_environment.yml
conda activate RFDpoly_env
pip install /path/to/RFDpoly/rf_diffusion/RF2-allatom/rf2aa/SE3Transformer
```
The last command installs a slightly modified version of 
se3-transformer. If installing this package does not work,
you can try installing [`se3-transformer-pytorch`](https://pypi.org/project/se3-transformer-pytorch/) instead,
but we do not guarantee that your results will be exactly
the same. 

You can test your installation by running: 
```bash
python /path/to/RFDpoly/rf_diffusion/run_inference.py \
--config-name=multi_polymer \
diffuser.T=50 \
inference.ckpt_path=/path/to/RFDpoly/weights/train_session2024-07-08_1720455712_BFF_3.00.pt \
inference.num_designs=1 \
contigmap.contigs=[\'33\ 33\ 75\'] \
contigmap.polymer_chains=[\'dna\',\'rna\',\'protein\'] \
inference.output_prefix=./basic_uncond_test01
```

If using the `macos_environment.yml` file does not work, 
please create an issue. Or, if you modified the `.yml` file 
to work for your system, make a pull request to add it to 
the `rf_diffusion/environment` folder. 
