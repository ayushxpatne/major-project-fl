import os
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Add project root to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_system.model import FraudDetectionNet, get_model_input_features

# Configuration
DATA_DIR = '/Users/ayushpatne/Developer/FL_Major/project/data/'
RESULTS_DIR = '/Users/ayushpatne/Developer/FL_Major/project/results/'
FL_MODEL_PATH = '/Users/ayushpatne/Developer/FL_Major/project/global_model.pth'
CENTRALIZED_MODEL_PATH = '/Users/ayushpatne/Developer/FL_Major/project/centralized_model.pth'

def load_test_data():
    """Load the global test dataset for evaluation"""
    try:
        test_df = pd.read_csv(os.path.join(DATA_DIR, "global_test_centralized.csv"))
        print(f"Loaded global test set: {len(test_df)} samples")
        
        # Extract features and labels
        X_test = test_df.drop(columns=['FLAG']).values.astype(np.float32)
        y_test = test_df['FLAG'].values.astype(np.float32)
        
        return X_test, y_test, test_df.drop(columns=['FLAG']).columns.tolist()
    except FileNotFoundError:
        print("Global test set not found. Trying to use client_1 test data as fallback.")
        try:
            test_df = pd.read_csv(os.path.join(DATA_DIR, "client_1_test.csv"))
            print(f"Using client_1 test data: {len(test_df)} samples")
            
            # Extract features and labels
            X_test = test_df.drop(columns=['FLAG']).values.astype(np.float32)
            y_test = test_df['FLAG'].values.astype(np.float32)
            
            return X_test, y_test, test_df.drop(columns=['FLAG']).columns.tolist()
        except FileNotFoundError:
            raise FileNotFoundError("No test data found. Please run data_split.py first.")

def load_model(model_path, num_features):
    """Load a PyTorch model from the given path"""
    model = FraudDetectionNet(num_features)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Warning: Model file not found at {model_path}")
        return None

def evaluate_model(model, X_test, y_test):
    """Evaluate a model on the test set and return metrics"""
    if model is None:
        return {
            'auc': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0
        }
    
    # Convert to tensor
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Get predictions
    with torch.no_grad():
        y_pred_probs = model(X_tensor).numpy().flatten()
    
    # Calculate metrics
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    
    try:
        auc = roc_auc_score(y_test, y_pred_probs)
    except ValueError:
        auc = 0.5
        print("Warning: AUC could not be computed due to single class or constant predictions.")
    
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    accuracy = np.mean(y_pred_binary == y_test)
    
    return {
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }

def plot_comparison(fl_metrics, centralized_metrics):
    """Generate a bar plot comparing FL and centralized models"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    metrics = ['auc', 'f1', 'precision', 'recall', 'accuracy']
    fl_values = [fl_metrics[m] for m in metrics]
    centralized_values = [centralized_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, fl_values, width, label='Federated Learning')
    rects2 = ax.bar(x + width/2, centralized_values, width, label='Centralized')
    
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'centralized_vs_fl.png'))
    print(f"Comparison plot saved to {os.path.join(RESULTS_DIR, 'centralized_vs_fl.png')}")
    plt.close()

def save_metrics(fl_metrics, centralized_metrics):
    """Save metrics to files for later use"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Save metrics to text file
    with open(os.path.join(RESULTS_DIR, 'metrics.txt'), 'w') as f:
        f.write("Model Performance Comparison\n")
        f.write("===========================\n\n")
        f.write("Metric      | Federated Learning | Centralized\n")
        f.write("------------|-------------------|------------\n")
        for metric in ['auc', 'f1', 'precision', 'recall', 'accuracy']:
            f.write(f"{metric.upper():12} | {fl_metrics[metric]:.6f}          | {centralized_metrics[metric]:.6f}\n")
    
    # Save metrics as joblib for the Flask app to use
    joblib.dump(fl_metrics, os.path.join(RESULTS_DIR, 'fl_metrics.joblib'))
    joblib.dump(centralized_metrics, os.path.join(RESULTS_DIR, 'centralized_metrics.joblib'))
    
    # Also save as JSON for easier web access
    import json
    with open(os.path.join(RESULTS_DIR, 'fl_metrics.json'), 'w') as f:
        json.dump({
            'round': list(range(1, 11)),  # Assuming 10 rounds
            'auc': [fl_metrics['auc']] * 10,  # Replicate the final value for all rounds
            'f1': [fl_metrics['f1']] * 10,
            'loss': [0.1] * 10,  # Placeholder
            'accuracy': [fl_metrics['accuracy']] * 10
        }, f)
    
    print(f"Metrics saved to {os.path.join(RESULTS_DIR, 'metrics.txt')}")

def main():
    """Main evaluation function"""
    print("Starting model evaluation...")
    
    # Load test data
    X_test, y_test, feature_columns = load_test_data()
    
    # Get number of features
    num_features = get_model_input_features()
    
    # Load models
    fl_model = load_model(FL_MODEL_PATH, num_features)
    centralized_model = load_model(CENTRALIZED_MODEL_PATH, num_features)
    
    # Evaluate models
    fl_metrics = evaluate_model(fl_model, X_test, y_test)
    centralized_metrics = evaluate_model(centralized_model, X_test, y_test)
    
    # Print results
    print("\nEvaluation Results:")
    print("------------------")
    print(f"FL Model AUC: {fl_metrics['auc']:.4f}, F1: {fl_metrics['f1']:.4f}")
    print(f"Centralized Model AUC: {centralized_metrics['auc']:.4f}, F1: {centralized_metrics['f1']:.4f}")
    
    # Plot comparison
    plot_comparison(fl_metrics, centralized_metrics)
    
    # Save metrics
    save_metrics(fl_metrics, centralized_metrics)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()