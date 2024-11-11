# btc_lab_image_analysis

## Usage

For some of the code to run (namely stain normalization), a virtual environment with an older version of python is required (python 3.10.5). 

Please activate a conda environment:

```bash
conda create -c conda-forge -n btc-labs python=3.10.5
```

Note: to activate an environment with conda, you need to already be in the conda base environment. This is usually done by default, but you may need to run the `source activate base` before running the code below

```bash
conda activate btc-labs
```

You will now be able so install the right packages to run all files

```bash
pip install -r requirements.txt
```