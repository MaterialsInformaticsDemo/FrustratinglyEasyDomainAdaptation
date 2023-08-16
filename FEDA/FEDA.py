# Frustratingly Easy Domain Adaptation
# cao bin, HKUST, China, binjacobcao@gmail.com
# free to charge for academic communication

import numpy as np
import pandas as pd 

def FEDA_features(source_file, target_file):
    """
    Generate feature-expanded datasets for domain adaptation.
    
    Args:
        source_file (str): Path to the CSV file containing source data.
        target_file (str): Path to the CSV file containing target data.
        
    Returns:
        pd.DataFrame, pd.DataFrame: Feature-expanded datasets for source and target.
    """
    # Read source data from CSV file
    source_data = pd.read_csv(source_file)
    target_data = pd.read_csv(target_file)
    source_header = source_data.columns.tolist()
    target_header = target_data.columns.tolist()
    x_source = np.array(source_data.iloc[:,:-1])
    y_source = np.array(source_data.iloc[:,-1])
    x_target = np.array(target_data.iloc[:,:-1])
    y_target = np.array(target_data.iloc[:,-1])
 
    # Expand features for source and target
    x_source_expand = np.hstack([x_source, x_source, np.zeros_like(x_source)])
    x_target_expand = np.hstack([x_target, np.zeros_like(x_target), x_target])

    # Generate expanded header
    expand_header = []
    for name in source_header[:-1]:
        expand_header.append(name + '_g')
    for name in source_header[:-1]:
        expand_header.append(name + '_s')
    for name in target_header[:-1]:
        expand_header.append(name + '_t')
    
    # Create DataFrames for expanded data
    data_source_expand = pd.DataFrame(x_source_expand, columns=expand_header)
    data_target_expand = pd.DataFrame(x_target_expand, columns=expand_header)

    # Add labels to the DataFrames
    data_source_expand[source_header[-1]] = np.array(y_source)
    data_target_expand[target_header[-1]] = np.array(y_target)

    return data_source_expand, data_target_expand
