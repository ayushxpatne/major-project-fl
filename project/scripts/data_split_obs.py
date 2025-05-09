import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATASET_PATH = '/Users/ayushpatne/Developer/FL_Major/transaction_dataset.csv'
OUTPUT_DATA_DIR = '/Users/ayushpatne/Developer/FL_Major/project/data/'
NUM_CLIENTS = 3
# Client fraud ratios: Client 1: 5%, Client 2: 10%, Client 3: 20%
# These are target fraud ratios for the *entirety* of each client's data (train+test)
# The actual split might vary slightly due to sampling.
CLIENT_FRAUD_CONFIG = {
    0: {'fraud_ratio': 0.05, 'name': 'client_1'},
    1: {'fraud_ratio': 0.10, 'name': 'client_2'},
    2: {'fraud_ratio': 0.20, 'name': 'client_3'}
}
TEST_SIZE = 0.2 # Test set size for each client's data
RANDOM_STATE = 42

def write_to_exploration_file(message):
    """
    Writes/appends messages to the exploration_cleaning.txt file in the data directory.
    
    Args:
        message (str): The message to write to the file
    """
    exploration_file = os.path.join(OUTPUT_DATA_DIR, 'exploration_cleaning.txt')
    
    # Create data directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    # Append message to file with timestamp
    with open(exploration_file, 'a') as f:
        # timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'[{timestamp}] {message}\n')


def load_and_preprocess_data(file_path):
    """Loads and preprocesses the Ethereum fraud dataset."""
    df = pd.read_csv(file_path)
    df.set_index('Index', inplace=True)
    
    df.sort_index()

    df['FLAG'].value_counts().plot(kind='bar')

    title = "Flag Distribution Before Preprocessing"
    plt.title(title)
    plt.savefig(os.path.join(OUTPUT_DATA_DIR, f"{title.replace(' ', '_').lower()}.png"))

    # Strip spaces9 from feature names
    df.columns = df.columns.str.strip()

    # Drop 'Unnamed: 0' since this feature doesn't provide any useful data
    # Drop 'Address' since this feature provides a unique ERC20 address
    # Drop 'Index' since its not unique, therefore useless

    df = df.drop(columns=['Unnamed: 0', 'Address', 'Index'], errors='ignore')  
    # Check for missing values and data types
    missing_values = df.isnull().sum()
    missing_percent = missing_values / len(df) * 100
    data_types = df.dtypes

    # Print missing values percentage per feature
    # print(missing_percent[missing_percent > 0], data_types)

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns

    # Check the amount of classes in our categorical features

    df['ERC20 most sent token type'].value_counts().plot(kind='bar')
    title = "ERC20 most sent token type Distribution"
    plt.title(title)
    plt.savefig(os.path.join(OUTPUT_DATA_DIR, f"{title.replace(' ', '_').lower()}.png"))

    df['ERC20_most_rec_token_type'].value_counts().plot(kind='bar')
    title = "ERC20 most rec token type Distribution"
    plt.title(title)
    plt.savefig(os.path.join(OUTPUT_DATA_DIR, f"{title.replace(' ', '_').lower()}.png"))

    # Plot distributions for every numerical feature
    for col in numerical_cols:
        print(f"Distribution of {col}:")
        title = f'plot distributions for {col} numerical feature'
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(title)
        plt.savefig(os.path.join(OUTPUT_DATA_DIR, f"{title.replace(' ', '_').lower()}.png"))
        # plt.show()

    no_var = df[numerical_cols].var() == 0
    no_var[no_var].index.tolist()

    # Replace empty columns by zeros
    df[numerical_cols] = df[numerical_cols].fillna(0)  # Assume missing = no ERC20 activity
    # Drop categorical columns since they may drastically increase the complexity of our model without providing significantly useful information.
    # More than a half is equal to '0' or ' '
    df = df.drop(columns=categorical_cols)
    # Drop columns with no variance
    df = df.drop(columns=no_var[no_var].index)
    # Find remaining features with a small distribution
    for i in df.columns[1:]:
        if len(df[i].value_counts()) < 10:
            print(df[i].value_counts())
    df = df.drop(columns={'min value sent to contract','total ether sent contracts','ERC20 uniq sent addr.1'})
    # df.describe()

    # Step 1: Select numerical columns
    numerical_df = df.select_dtypes(include=['int64', 'float64'])

    # Step 2: Compute the correlation matrix
    correlation_matrix = numerical_df.corr()

    # Step 3: Plot the correlation matrix with gradient
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        correlation_matrix,
        cmap = "coolwarm",
        vmin=-1, vmax=1,
        linewidths=0,
    )
    title = "Correlation Matrix of Numerical Features"
    plt.title(title, fontsize=18)
    plt.savefig(os.path.join(OUTPUT_DATA_DIR, f"{title.replace(' ', '_').lower()}.png"))

    # Detecting highly correlated pairs from the correlation matrix above
    threshold = 0.85  # Set your correlation threshold

    # Create a list of pairs of highly correlated features
    highly_correlated_pairs = []

    # Loop through the correlation matrix
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                pair = (correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j])
                highly_correlated_pairs.append(pair)

    print("Highly Correlated Pairs:")
    # Display the list of highly correlated pairs
    for pair in highly_correlated_pairs:
        print(f"{pair[0]} + {pair[1]}: {pair[2]:.2f}")
    

    # Create a set of features to remove
    features_to_remove = {
        'avg value sent to contract',  
        'max val sent to contract',
        'ERC20 max val rec',
        'ERC20 avg val rec',
        'ERC20 min val sent',
        'ERC20 max val sent',
        'ERC20 avg val sent',
        'ERC20 uniq rec token name',
    }

    # Drop the highly correlated features
    df = df.drop(columns=features_to_remove)

    # Check the correlation matrix one more time after the cleaning is finished

    # Step 1: Select numerical columns
    numerical_df = df.select_dtypes(include=['int64', 'float64'])

    # Step 2: Compute the correlation matrix
    correlation_matrix = numerical_df.corr()

    # Step 3: Plot the correlation matrix with gradient
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        correlation_matrix,
        cmap = "coolwarm",
        vmin=-1, vmax=1,
        linewidths=0,
    )

    title = "Correlation Matrix of Numerical Features After Dropping HIghlighy Correlated "
    plt.title("Correlation Matrix of Numerical Features After Dropping HIghlighy Correlated ", fontsize=18)
    plt.savefig(os.path.join(OUTPUT_DATA_DIR, f"{title.replace(' ', '_').lower()}.png"))
    # plt.show()

load_and_preprocess_data(DATASET_PATH)


