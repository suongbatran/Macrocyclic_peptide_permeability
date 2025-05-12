import pandas as pd
import os
from sklearn.model_selection import train_test_split

def random_split_data(df, output_dir):
    
    train, temp = train_test_split(df, train_size= 0.8, random_state=42)
    val, test = train_test_split(temp, test_size= 1/2 , random_state=42)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save file
    train.to_csv(f"{output_dir}/raw_train.csv", index=False)
    val.to_csv(f"{output_dir}/raw_val.csv", index=False)
    test.to_csv(f"{output_dir}/raw_test.csv", index=False)

    print(f"Saved random split: {len(train)} train, {len(val)} val, {len(test)} test")

def stratified_monomer_split_data(df, output_dir, stratify_column='num_monomers'):
    # Stratified split based on the 'num_monomer' column to maintain even distribution of classes
    train, temp = train_test_split(df, train_size=0.8, stratify=df[stratify_column], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp[stratify_column], random_state=42)

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the split files
    train.to_csv(f"{output_dir}/raw_train.csv", index=False)
    val.to_csv(f"{output_dir}/raw_val.csv", index=False)
    test.to_csv(f"{output_dir}/raw_test.csv", index=False)

    print(f"Saved stratified random split of num_monomers: {len(train)} train, {len(val)} val, {len(test)} test")

def num_monomer_split_data(df, output_dir):

    # Train set
    train = df[df['num_monomers'] == 6]

    # Validation and test set
    group_seven = df[df['num_monomers'] == 7]
    group_seven = group_seven.sample(frac=1, random_state=42) 
    val = group_seven.iloc[:len(group_seven)//2]
    test = pd.concat([group_seven.iloc[len(group_seven)//2:], df[df['num_monomers'] == 10]])

    # Save file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    train.to_csv(f"{output_dir}/raw_train.csv", index=False)
    val.to_csv(f"{output_dir}/raw_val.csv", index=False)
    test.to_csv(f"{output_dir}/raw_test.csv", index=False)

    print(f"Saved monomer splits: {len(train)} train, {len(val)} val, {len(test)} test")

###### Main

input_file = "data/permeability.csv"
df = pd.read_csv(input_file)

df = df[['CycPeptMPDB_ID', 'sequence', 'permeability', 'num_monomers']]
df = df[df['permeability'] != -10 ]

# Random split
random_output_dir = "data/split_random/raw"
random_split_data(df, random_output_dir)

# Split by number of monomer
monomer_output_dir = "data/split_monomer/raw"
num_monomer_split_data(df, monomer_output_dir)

# Stratified split by number of monomer
strat_monomer_output_dir = "data/strat_split_monomer/raw"
stratified_monomer_split_data(df, strat_monomer_output_dir)
