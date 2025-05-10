import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuration
TRAIN_DATA_PATH = '/Users/ayushpatne/Developer/FL_Major/project/data/global_train_centralized.csv'
TEST_DATA_PATH = '/Users/ayushpatne/Developer/FL_Major/project/data/global_test_centralized.csv'
OUTPUT_DATA_DIR = '/Users/ayushpatne/Developer/FL_Major/project/data/'
NUM_CLIENTS = 3

# Client fraud ratios configuration
CLIENT_FRAUD_CONFIG = {
    0: {'fraud_ratio': 0.04, 'name': 'client_1'},
    1: {'fraud_ratio': 0.06, 'name': 'client_2'},
    2: {'fraud_ratio': 0.08, 'name': 'client_3'}
}
RANDOM_STATE = 42

def create_non_iid_splits(train_data, test_data, num_clients, client_fraud_config, random_state):
    """Creates non-IID data splits for clients based on fraud ratios using pre-split data."""
    
    # Process training data
    train_fraud = train_data[train_data['FLAG'] == 1]
    train_non_fraud = train_data[train_data['FLAG'] == 0]
    
    # Process test data
    test_fraud = test_data[test_data['FLAG'] == 1]
    test_non_fraud = test_data[test_data['FLAG'] == 0]
    
    # Shuffle data
    train_fraud = train_fraud.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_non_fraud = train_non_fraud.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_fraud = test_fraud.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_non_fraud = test_non_fraud.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    client_datasets = []
    
    # Calculate approximate samples per client
    train_samples_per_client = len(train_data) // num_clients
    test_samples_per_client = len(test_data) // num_clients
    
    train_fraud_idx = 0
    train_non_fraud_idx = 0
    test_fraud_idx = 0
    test_non_fraud_idx = 0
    
    for client_id in range(num_clients):
        target_fraud_ratio = client_fraud_config[client_id]['fraud_ratio']
        
        # Calculate samples for training set
        train_fraud_samples = int(train_samples_per_client * target_fraud_ratio)
        train_non_fraud_samples = train_samples_per_client - train_fraud_samples
        
        # Calculate samples for test set
        test_fraud_samples = int(test_samples_per_client * target_fraud_ratio)
        test_non_fraud_samples = test_samples_per_client - test_fraud_samples
        
        # Create client training set
        client_train_fraud = train_fraud.iloc[train_fraud_idx:train_fraud_idx + train_fraud_samples]
        client_train_non_fraud = train_non_fraud.iloc[train_non_fraud_idx:train_non_fraud_idx + train_non_fraud_samples]
        client_train = pd.concat([client_train_fraud, client_train_non_fraud]).sample(frac=1, random_state=random_state)
        
        # Create client test set
        client_test_fraud = test_fraud.iloc[test_fraud_idx:test_fraud_idx + test_fraud_samples]
        client_test_non_fraud = test_non_fraud.iloc[test_non_fraud_idx:test_non_fraud_idx + test_non_fraud_samples]
        client_test = pd.concat([client_test_fraud, client_test_non_fraud]).sample(frac=1, random_state=random_state)
        
        # Update indices
        train_fraud_idx += train_fraud_samples
        train_non_fraud_idx += train_non_fraud_samples
        test_fraud_idx += test_fraud_samples
        test_non_fraud_idx += test_non_fraud_samples
        
        client_datasets.append({
            'name': client_fraud_config[client_id]['name'],
            'train': client_train,
            'test': client_test
        })
        
        # Create client directory and save visualizations
        client_output_dir = os.path.join(OUTPUT_DATA_DIR, client_fraud_config[client_id]['name'])
        if not os.path.exists(client_output_dir):
            os.makedirs(client_output_dir)
            
        # Plot distributions and save statistics
        plot_data_info(client_train, client_output_dir)
        plot_class_distribution(client_train['FLAG'], "Train Class Distribution", client_output_dir)
        plot_class_distribution(client_test['FLAG'], "Test Class Distribution", client_output_dir)
        
        # Print client statistics
        actual_fraud_ratio_train = client_train['FLAG'].mean()
        actual_fraud_ratio_test = client_test['FLAG'].mean()
        print(f"Client {client_id} ({client_fraud_config[client_id]['name']}): "
              f"Train size={len(client_train)}, Test size={len(client_test)}, "
              f"Train Fraud Ratio={actual_fraud_ratio_train:.3f}, Test Fraud Ratio={actual_fraud_ratio_test:.3f}")
    
    return client_datasets

def save_client_data(client_datasets, output_dir):
    """Saves client datasets to CSV files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for client_data in client_datasets:
        client_name = client_data['name']
        client_data['train'].to_csv(os.path.join(output_dir, f"{client_name}_train.csv"), index=False)
        client_data['test'].to_csv(os.path.join(output_dir, f"{client_name}_test.csv"), index=False)
        print(f"Saved data for {client_name}")

def plot_data_info(df, output_dir):
    """Saves a summary of the dataset information."""
    info_file = os.path.join(output_dir, "data_info.txt")
    with open(info_file, "w") as f:
        df.info(buf=f)

def plot_class_distribution(y, title, output_dir):
    """Plots the distribution of classes in the dataset."""
    plt.figure(figsize=(8, 6))
    y.value_counts().plot(kind='bar')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').lower()}_distribution.png"))
    plt.close()

if __name__ == "__main__":
    # Load pre-processed and split data
    print("Loading pre-processed train and test data...")
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)
    
    print("\nCreating non-IID client splits...")
    client_datasets = create_non_iid_splits(train_data, test_data, NUM_CLIENTS, CLIENT_FRAUD_CONFIG, RANDOM_STATE)
    
    print("\nSaving client data...")
    save_client_data(client_datasets, OUTPUT_DATA_DIR)
    
    # Save feature columns
    feature_columns = train_data.columns.drop('FLAG').tolist()
    joblib.dump(feature_columns, os.path.join(OUTPUT_DATA_DIR, 'feature_columns.joblib'))
    print("\nSaved feature columns to feature_columns.joblib")
    
    print("\nData splitting process complete.")
