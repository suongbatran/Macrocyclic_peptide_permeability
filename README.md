## Predicting Membrane Permeability of Macrocyclic Peptides

## Load environment 

```bash
conda env create -f environment.yml
conda activate perpeptide
```
For Graph 3D SchNet:
```bash
conda env create -f environment_schnet.yml
conda activate schnet
```
## Data
#### Download data

```bash
mkdir -p data
wget https://zenodo.org/records/13334335/files/summary_cycpeptmpdb.csv?download=1 -O data/permeability.csv
wget https://zenodo.org/records/13334335/files/pickle.tar.gz?download=1 -O data/pickle.tar.gz
mkdir -p data/pickle && tar -xvzf pickle.tar.gz -C data/pickle
rm data/pickle.tar.gz
```

#### Split data for model training and testing

```bash
python scripts/train_val_test_split.py
```

Each file in the “pickle” folder contains a Python dictionary with amino acid sequence, SMILES, CREST metadata, and a single RDKit molecule object containing all conformers.

Each `sequence.pickle` in the `pickle` folder is has the name of the `sequence` column provided in permeability
You could use `view_data.ipynb` to see how the data is stored.

## Retrain models 

To re-train the models, follow the steps above to set up the environment and acquire the dataset then:

- update the split dataset as ```random split```, ```monomer-based split``` , or ```stratified monomer-based split``` in ```benchmark.sh``` file in each model folder

```bash
export DATA_NAME="random split"
```
- replace MODEL_PATH name and run the following command:

```bash
export MODEL_PATH="Graph 2D"
sh scripts/model/$MODEL_PATH/benchmark.sh
```

