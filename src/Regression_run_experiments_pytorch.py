import os
import sys
from collections import defaultdict
import warnings
import logging
from typing import Literal
from sklearn.model_selection import KFold, GroupKFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import protac_degradation_predictor as pdp

import pytorch_lightning as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from jsonargparse import CLI
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
)

# Ignore UserWarning from Matplotlib
warnings.filterwarnings("ignore", ".*FixedLocator*")
# Ignore UserWarning from PyTorch Lightning
warnings.filterwarnings("ignore", ".*does not have many workers.*")


root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


def get_random_split_indices(active_df: pd.DataFrame, test_split: float) -> pd.Index:
    """ Get the indices of the test set using a random split.
    
    Args:
        active_df (pd.DataFrame): The DataFrame containing the active PROTACs.
        test_split (float): The percentage of the active PROTACs to use as the test set.
    
    Returns:
        pd.Index: The indices of the test set.
    """
    test_df = active_df.sample(frac=test_split, random_state=42)
    return test_df.index


def get_e3_ligase_split_indices(active_df: pd.DataFrame) -> pd.Index:
    """ Get the indices of the test set using the E3 ligase split.
    
    Args:
        active_df (pd.DataFrame): The DataFrame containing the active PROTACs.
    
    Returns:
        pd.Index: The indices of the test set.
    """
    encoder = OrdinalEncoder()
    active_df['E3 Group'] = encoder.fit_transform(active_df[['E3 Ligase']]).astype(int)
    test_df = active_df[(active_df['E3 Ligase'] != 'VHL') & (active_df['E3 Ligase'] != 'CRBN')]
    return test_df.index


def get_smiles2fp_and_avg_tanimoto(protac_df: pd.DataFrame) -> tuple:
    """ Get the SMILES to fingerprint dictionary and the average Tanimoto similarity.
    
    Args:
        protac_df (pd.DataFrame): The DataFrame containing the PROTACs.
    
    Returns:
        tuple: The SMILES to fingerprint dictionary and the average Tanimoto similarity.
    """
    unique_smiles = protac_df['Smiles'].unique().tolist()

    smiles2fp = {}
    for smiles in tqdm(unique_smiles, desc='Precomputing fingerprints'):
        smiles2fp[smiles] = pdp.get_fingerprint(smiles)

    # # Get the pair-wise tanimoto similarity between the PROTAC fingerprints
    # tanimoto_matrix = defaultdict(list)
    # for i, smiles1 in enumerate(tqdm(protac_df['Smiles'].unique(), desc='Computing Tanimoto similarity')):
    #     fp1 = smiles2fp[smiles1]
    #     # TODO: Use BulkTanimotoSimilarity for better performance
    #     for j, smiles2 in enumerate(protac_df['Smiles'].unique()[i:]):
    #         fp2 = smiles2fp[smiles2]
    #         tanimoto_dist = 1 - DataStructs.TanimotoSimilarity(fp1, fp2)
    #         tanimoto_matrix[smiles1].append(tanimoto_dist)
    # avg_tanimoto = {k: np.mean(v) for k, v in tanimoto_matrix.items()}
    # protac_df['Avg Tanimoto'] = protac_df['Smiles'].map(avg_tanimoto)


    tanimoto_matrix = defaultdict(list)
    fps = list(smiles2fp.values())

    # Compute all-against-all Tanimoto similarity using BulkTanimotoSimilarity
    for i, (smiles1, fp1) in enumerate(tqdm(zip(unique_smiles, fps), desc='Computing Tanimoto similarity', total=len(fps))):
        similarities = DataStructs.BulkTanimotoSimilarity(fp1, fps[i:])  # Only compute for i to end, avoiding duplicates
        for j, similarity in enumerate(similarities):
            distance = 1 - similarity
            tanimoto_matrix[smiles1].append(distance)  # Store as distance
            if i != i + j:
                tanimoto_matrix[unique_smiles[i + j]].append(distance)  # Symmetric filling

    # Calculate average Tanimoto distance for each unique SMILES
    avg_tanimoto = {k: np.mean(v) for k, v in tanimoto_matrix.items()}
    protac_df['Avg Tanimoto'] = protac_df['Smiles'].map(avg_tanimoto)

    smiles2fp = {s: np.array(fp) for s, fp in smiles2fp.items()}

    return smiles2fp, protac_df


def get_tanimoto_split_indices(
        active_df: pd.DataFrame,
        active_col: str,
        test_split: float,
        n_bins_tanimoto: int = 200,
) -> pd.Index:
    """ Get the indices of the test set using the Tanimoto-based split.
    
    Args:
        active_df (pd.DataFrame): The DataFrame containing the active PROTACs.
        n_bins_tanimoto (int): The number of bins to use for the Tanimoto similarity.
    
    Returns:
        pd.Index: The indices of the test set.
    """
    tanimoto_groups = pd.cut(active_df['Avg Tanimoto'], bins=n_bins_tanimoto).copy()
    encoder = OrdinalEncoder()
    active_df['Tanimoto Group'] = encoder.fit_transform(tanimoto_groups.values.reshape(-1, 1)).astype(int)
    # Sort the groups so that samples with the highest tanimoto similarity,
    # i.e., the "less similar" ones, are placed in the test set first
    tanimoto_groups = active_df.groupby('Tanimoto Group')['Avg Tanimoto'].mean().sort_values(ascending=False).index

    test_df = []
    # For each group, get the number of active and inactive entries. Then, add those
    # entries to the test_df if: 1) the test_df lenght + the group entries is less
    # 20% of the active_df lenght, and 2) the percentage of True and False entries
    # in the active_col in test_df is roughly 50%.
    for group in tanimoto_groups:
        group_df = active_df[active_df['Tanimoto Group'] == group]
        if test_df == []:
            test_df.append(group_df)
            continue
        
        num_entries = len(group_df)
        num_active_group = group_df[active_col].sum()
        num_inactive_group = num_entries - num_active_group

        tmp_test_df = pd.concat(test_df)
        num_entries_test = len(tmp_test_df)
        num_active_test = tmp_test_df[active_col].sum()
        num_inactive_test = num_entries_test - num_active_test
        
        # Check if the group entries can be added to the test_df
        if num_entries_test + num_entries < test_split * len(active_df):
            # Add anything at the beggining
            if num_entries_test + num_entries < test_split / 2 * len(active_df):
                test_df.append(group_df)
                continue
            # Be more selective and make sure that the percentage of active and
            # inactive is balanced
            if (num_active_group + num_active_test) / (num_entries_test + num_entries) < 0.6:
                if (num_inactive_group + num_inactive_test) / (num_entries_test + num_entries) < 0.6:
                    test_df.append(group_df)
    test_df = pd.concat(test_df)
    return test_df.index


def get_target_split_indices(active_df: pd.DataFrame, active_col: str, test_split: float) -> pd.Index:
    """ Get the indices of the test set using the target-based split.

    Args:
        active_df (pd.DataFrame): The DataFrame containing the active PROTACs.
        active_col (str): The column containing the active/inactive information.
        test_split (float): The percentage of the active PROTACs to use as the test set.

    Returns:
        pd.Index: The indices of the test set.
    """
    encoder = OrdinalEncoder()
    active_df['Uniprot Group'] = encoder.fit_transform(active_df[['Uniprot']]).astype(int)

    test_df = []
    # For each group, get the number of active and inactive entries. Then, add those
    # entries to the test_df if: 1) the test_df lenght + the group entries is less
    # 20% of the active_df lenght, and 2) the percentage of True and False entries
    # in the active_col in test_df is roughly 50%.
    # Start the loop from the groups containing the smallest number of entries.
    for group in reversed(active_df['Uniprot'].value_counts().index):
        group_df = active_df[active_df['Uniprot'] == group]
        if test_df == []:
            test_df.append(group_df)
            continue
        
        num_entries = len(group_df)
        num_active_group = group_df[active_col].sum()
        num_inactive_group = num_entries - num_active_group

        tmp_test_df = pd.concat(test_df)
        num_entries_test = len(tmp_test_df)
        num_active_test = tmp_test_df[active_col].sum()
        num_inactive_test = num_entries_test - num_active_test
        
        # Check if the group entries can be added to the test_df
        if num_entries_test + num_entries < test_split * len(active_df):
            # Add anything at the beggining
            if num_entries_test + num_entries < test_split / 2 * len(active_df):
                test_df.append(group_df)
                continue
            # Be more selective and make sure that the percentage of active and
            # inactive is balanced
            if (num_active_group + num_active_test) / (num_entries_test + num_entries) < 0.6:
                if (num_inactive_group + num_inactive_test) / (num_entries_test + num_entries) < 0.6:
                    test_df.append(group_df)
    test_df = pd.concat(test_df)
    return test_df.index

def safe_log10(x):
    """turn DC50 to pDC50"""
    return np.log10(np.maximum(x, 1e-10))

## When regressio: normalize Dmax, TODO
def preprocess_dmax(df, column_name='Dmax (%)'):
    """
    Preprocess Dmax data by filtering, normalizing, and applying a logit transformation.

    Args:
        df : pandas.DataFrame
            DataFrame that contains Dmax data.
        column_name : str
            The name of the column to process (default: 'Dmax (%)').

    Returns:
        pandas.DataFrame
            The processed DataFrame with the logit-transformed column.
    """
    # 1. Remove rows where Dmax (%) is greater than 100
    df = df[df[column_name] <= 100]

    # 2. Convert Dmax (%) from percentage to a proportion (0-1)
    df[column_name] = df[column_name] / 100

    # 3. Handle edge cases for proportions equal to 0 or 1
    df[column_name] = df[column_name].apply(lambda x: 0.999 if x == 1 else (0.001 if x == 0 else x))

    # Apply logit transformation
    # logit_column_name = column_name + '_logit'
    df[column_name] = np.log(df[column_name] / (1 - df[column_name]))

    return df

def main(
    active_col: str = 'pDC50', # When regression, change to Dmax or pDC50
    
    n_trials: int = 200,## can be more, regression
    fast_dev_run: bool = False,
    test_split: float = 0.1,
    cv_n_splits: int = 5,
    max_epochs: int = 200,
    run_sklearn: bool = False,
    force_study: bool = True,
    experiments: str | Literal['all', 'standard', 'e3_ligase', 'similarity', 'target'] = 'all',
):
    """ Train a PROTAC model using the given datasets and hyperparameters.
    
    Args:
        active_col (str): The column containing the active/inactive information. Must be in the format 'Active (Dmax N, pDC50 M)'.
        n_trials (int): The number of hyperparameter tuning trials to run.
        fast_dev_run (bool): Whether to run a fast development run.
        test_split (float): The percentage of the active PROTACs to use as the test set.
        cv_n_splits (int): The number of cross-validation splits to use.
        max_epochs (int): The maximum number of epochs to train the model.
        run_sklearn (bool): Whether to run sklearn models.
        force_study (bool): Whether to force the creation of a new Optuna study.
        experiments (str): The type of experiments to run.
    """
    pl.seed_everything(42)


    # Make directory ../reports if it does not exist
    if not os.path.exists('../reports'):
        os.makedirs('../reports')

    # Load embedding dictionaries
    protein2embedding = pdp.load_protein2embedding('../data/uniprot2embedding.h5')
    cell2embedding = pdp.load_cell2embedding('../data/cell2embedding.pkl')

    studies_dir = '../data/studies'
    train_val_perc = f'{float((1 - test_split))}'
    test_perc = f'{float(test_split )}'
    #active_name = active_col.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')

    if experiments == 'all':
        experiments = ['standard', 'similarity', 'target']
    else:
        experiments = [experiments]

    # Cross-Validation Training
    reports = defaultdict(list)
    for split_type in experiments: 

        train_val_filename = f'DC50_{split_type}_train_{train_val_perc}.csv'
        test_filename = f'DC50_{split_type}_test_{test_perc}.csv'
        
        train_val_df = pd.read_csv(os.path.join(studies_dir, train_val_filename))
        test_df = pd.read_csv(os.path.join(studies_dir, test_filename))
        # train_val_df= preprocess_dmax(train_val_df,'Dmax (%)')
        # test_df= preprocess_dmax(test_df,'Dmax (%)')
        train_val_df['DC50 (nM)'] = train_val_df['DC50 (nM)'].replace(0, np.nan)  # Replace 0 values in DC50 (nM) with NaN
        train_val_df['pDC50'] = -safe_log10(train_val_df['DC50 (nM)'] * 1e-9)

        # test_df, same treatment
        test_df['DC50 (nM)'] = test_df['DC50 (nM)'].replace(0, np.nan)  # Replace 0 values in DC50 (nM) with NaN
        test_df['pDC50'] = -safe_log10(test_df['DC50 (nM)'] * 1e-9)

        # Replace infinite values with NaN
        train_val_df['pDC50'] = train_val_df['pDC50'].replace([np.inf, -np.inf], np.nan)
        test_df['pDC50'] = test_df['pDC50'].replace([np.inf, -np.inf], np.nan)

        # Drop rows with NaN in the target column
        train_val_df.dropna(subset=['pDC50'], inplace=True)
        test_df.dropna(subset=['pDC50'], inplace=True)
        print('lenth',len(train_val_df),len(test_df))
        # train_val_df['pDC50'] = train_val_df['pDC50'] / 100
        # test_df['pDC50'] = test_df['pDC50'] / 100
        # Get SMILES and precompute fingerprints dictionary
        unique_smiles = pd.concat([train_val_df, test_df])['Smiles'].unique().tolist()
        smiles2fp = {s: np.array(pdp.get_fingerprint(s)) for s in unique_smiles}

        if split_type == 'standard':
            kf = KFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = None
        elif split_type == 'e3_ligase':
            kf = GroupKFold(n_splits=cv_n_splits)  # GroupKFold does not support `shuffle`
            group = train_val_df['E3 Group'].to_numpy()
        elif split_type == 'similarity':
            kf = GroupKFold(n_splits=cv_n_splits)
            group = train_val_df['Tanimoto Group'].to_numpy()
        elif split_type == 'target':
            kf = GroupKFold(n_splits=cv_n_splits)
            group = train_val_df['Uniprot Group'].to_numpy()  

        # Start the experiment
        experiment_name = f'DC50_200_{split_type}_regression_test_split_{test_split}'
        optuna_reports = pdp.hyperparameter_tuning_and_training( 
            protein2embedding=protein2embedding,
            cell2embedding=cell2embedding,
            smiles2fp=smiles2fp,
            train_val_df=train_val_df,
            test_df=test_df,
            kf=kf,
            groups=group,
            split_type=split_type,
            n_models_for_test=3,
            fast_dev_run=fast_dev_run,
            n_trials=n_trials,
            max_epochs=max_epochs,
            logger_save_dir='../logs',
            logger_name=f'pytorch_{experiment_name}',
            active_label='pDC50',
            #regression_label= Dmax,
            study_filename=f'../reports/study_pytorch_{experiment_name}.pkl',
            force_study=force_study,
        )

        # Save the reports to file
        for report_name, report in optuna_reports.items():
            report.to_csv(f'../reports/pytorch_{report_name}_{experiment_name}.csv', index=False)
            reports[report_name].append(report.copy())


if __name__ == '__main__':
    cli = CLI(main)