import os
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve, confusion_matrix, auc
from collections import OrderedDict

# Add project root to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_system.model import FraudDetectionNet, get_model_input_features

# Configuration - Use relative paths for better portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FL_MODEL_PATH = os.path.join(BASE_DIR, 'global_model.pth')
CENTRALIZED_MODEL_PATH = os.path.join(BASE_DIR, 'centralized_model.pth')

# Constants for visualization
AGGREGATE_EVERY_N_ROUNDS = 3  # How often to save metrics and plots
NUM_ROUNDS = 10  # Total number of FL rounds

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

def train_centralized_model(X_train, y_train, num_features):
    """Train a centralized model on combined data"""
    model = FraudDetectionNet(num_features)
    
    # Convert to tensor
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    
    # Training parameters - match FL client settings
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_size = 32
    epochs = 10  # Slightly more than local FL client epochs to ensure fair comparison
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        # Simple batch implementation
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Centralized Training Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/(len(X_tensor)//batch_size):.4f}")
    
    return model

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
    """Evaluate a model on test data and return metrics"""
    if model is None:
        return {
            'auc': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0,
            'loss': 0.0,
            'predictions': None,
            'probabilities': None
        }
    
    # Convert to tensor
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    
    # Evaluate
    criterion = torch.nn.BCELoss()
    model.eval()
    with torch.no_grad():
        # Get model predictions
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor).item()
        
        # Convert to numpy for sklearn metrics
        y_true = y_test
        y_prob = outputs.numpy().flatten()
        y_pred = (y_prob > 0.5).astype(int)
        
        # Calculate metrics
        try:
            auc_score = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_score = 0.5
            print("Warning: AUC could not be computed due to single class or constant predictions.")
        
        accuracy = np.mean(y_pred == y_true)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
    # Return all metrics
    return {
        'auc': float(auc_score),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'loss': float(loss),
        'predictions': y_pred,
        'probabilities': y_prob
    }

def plot_roc_curve(y_true, y_prob_fl, y_prob_centralized=None):
    """Plot ROC curve for model(s)"""
    plt.figure(figsize=(10, 8))
    
    # Plot FL model ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob_fl)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'FL Model (AUC = {roc_auc:.3f})')
    
    # Plot centralized model ROC curve if available
    if y_prob_centralized is not None:
        fpr_c, tpr_c, _ = roc_curve(y_true, y_prob_centralized)
        roc_auc_c = auc(fpr_c, tpr_c)
        plt.plot(fpr_c, tpr_c, color='red', lw=2, label=f'Centralized (AUC = {roc_auc_c:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save the plot
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(y_true, y_prob_fl, y_prob_centralized=None):
    """Plot Precision-Recall curve for model(s)"""
    plt.figure(figsize=(10, 8))
    
    # Plot FL model PR curve
    precision_fl, recall_fl, _ = precision_recall_curve(y_true, y_prob_fl)
    pr_auc_fl = auc(recall_fl, precision_fl)
    plt.plot(recall_fl, precision_fl, color='blue', lw=2, label=f'FL Model (AUC = {pr_auc_fl:.3f})')
    
    # Plot centralized model PR curve if available
    if y_prob_centralized is not None:
        precision_c, recall_c, _ = precision_recall_curve(y_true, y_prob_centralized)
        pr_auc_c = auc(recall_c, precision_c)
        plt.plot(recall_c, precision_c, color='red', lw=2, label=f'Centralized (AUC = {pr_auc_c:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    
    # Save the plot
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    plt.savefig(os.path.join(RESULTS_DIR, 'precision_recall_curve.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred_fl, y_pred_centralized=None):
    """Plot confusion matrix for model(s)"""
    fig, axes = plt.subplots(1, 2 if y_pred_centralized is not None else 1, figsize=(16, 6))
    
    # FL model confusion matrix
    cm_fl = confusion_matrix(y_true, y_pred_fl)
    if y_pred_centralized is None:
        ax = axes
    else:
        ax = axes[0]
    
    im = ax.imshow(cm_fl, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("FL Model Confusion Matrix")
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add labels
    classes = ['Normal (0)', 'Fraud (1)']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    # Add text annotations
    thresh = cm_fl.max() / 2.
    for i in range(cm_fl.shape[0]):
        for j in range(cm_fl.shape[1]):
            ax.text(j, i, format(cm_fl[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_fl[i, j] > thresh else "black")
    
    # Centralized model confusion matrix if available
    if y_pred_centralized is not None:
        cm_c = confusion_matrix(y_true, y_pred_centralized)
        ax = axes[1]
        im = ax.imshow(cm_c, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title("Centralized Model Confusion Matrix")
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add labels
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        
        # Add text annotations
        thresh = cm_c.max() / 2.
        for i in range(cm_c.shape[0]):
            for j in range(cm_c.shape[1]):
                ax.text(j, i, format(cm_c[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm_c[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plt.close()

def plot_privacy_vs_utility(noise_levels, utility_metrics):
    """Plot privacy-utility tradeoff
    
    Args:
        noise_levels: List of noise levels used in DP
        utility_metrics: Dict with keys as metric names and values as lists of metrics per noise level
    """
    plt.figure(figsize=(12, 8))
    
    for metric_name, metric_values in utility_metrics.items():
        plt.plot(noise_levels, metric_values, marker='o', label=metric_name)
    
    plt.xlabel('Noise Scale (ε⁻¹)')
    plt.ylabel('Metric Value')
    plt.title('Privacy-Utility Tradeoff')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    plt.savefig(os.path.join(RESULTS_DIR, 'privacy_utility_tradeoff.png'))
    plt.close()

def plot_client_contributions(client_metrics):
    """Plot client contributions to the FL model
    
    Args:
        client_metrics: Dict with keys as client IDs and values as dicts of metrics
    """
    metrics = ['auc', 'f1', 'precision', 'recall', 'accuracy']
    clients = list(client_metrics.keys())
    
    plt.figure(figsize=(15, 10))
    
    width = 0.15
    x = np.arange(len(metrics))
    
    for i, client_id in enumerate(clients):
        values = [client_metrics[client_id][m] for m in metrics]
        plt.bar(x + i*width, values, width, label=f'Client {client_id}')
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Performance by Client')
    plt.xticks(x + width*(len(clients)-1)/2, [m.upper() for m in metrics])
    plt.legend(loc='best')
    plt.ylim(0, 1.0)
    
    # Save the plot
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    plt.savefig(os.path.join(RESULTS_DIR, 'client_contributions.png'))
    plt.close()
# Add this helper function somewhere in evaluate.py (e.g., at the top with other helper functions)
def convert_numpy_to_python(obj):
    """
    Recursively converts numpy types within an object to native Python types.
    This is necessary because json.dump cannot serialize numpy.ndarray or numpy.float64 etc.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Convert numpy arrays to Python lists
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(elem) for elem in obj]
    else:
        return obj

def save_full_comparison(fl_metrics, centralized_metrics):
    """Save comprehensive comparison metrics to files"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Save metrics to detailed text file
    with open(os.path.join(RESULTS_DIR, 'detailed_metrics.txt'), 'w') as f:
        f.write("===============================================\n")
        f.write("Federated Learning vs. Centralized Comparison\n")
        f.write("===============================================\n\n")
        
        f.write("1. CLASSIFICATION METRICS\n")
        f.write("------------------------\n")
        for metric in ['auc', 'f1', 'precision', 'recall', 'accuracy']:
            f.write(f"{metric.upper():12} | FL: {fl_metrics[metric]:.6f} | Centralized: {centralized_metrics[metric]:.6f} | Diff: {fl_metrics[metric] - centralized_metrics[metric]:.6f}\n")
        
        f.write("\n2. TRAINING EFFICIENCY\n")
        f.write("---------------------\n")
        f.write(f"Loss              | FL: {fl_metrics['loss']:.6f} | Centralized: {centralized_metrics['loss']:.6f}\n")
        
        f.write("\n3. PRIVACY ANALYSIS\n")
        f.write("------------------\n")
        f.write("Privacy Budget (ε): Not formally tracked in current implementation\n")
        f.write("Noise Scale: 0.01 (fixed in current implementation)\n")
        f.write("Clipping Threshold: 1.0 (fixed in current implementation)\n")
        
        f.write("\n4. SYSTEM PERFORMANCE\n")
        f.write("--------------------\n")
        f.write("Communication Rounds: 10\n")
        f.write("Clients: 3\n")
        f.write("Aggregation Strategy: FedAvg\n")

   

    
    # Save as JSON for easier programmatic access
    import json
    fl_metrics_serializable = convert_numpy_to_python(fl_metrics)
    centralized_metrics_serializable = convert_numpy_to_python(centralized_metrics)
    with open(os.path.join(RESULTS_DIR, 'comparison_metrics.json'), 'w') as f:
        json.dump({
            'fl': fl_metrics_serializable,
            'centralized': centralized_metrics_serializable,
            'system': {
                'rounds': NUM_ROUNDS,
                'clients': 3,
                'aggregation': 'FedAvg',
                'privacy': {
                    'noise_scale': 0.01,
                    'clipping_threshold': 1.0
                }
            }
        }, f, indent=2)
    
    print(f"Detailed comparison saved to {os.path.join(RESULTS_DIR, 'detailed_metrics.txt')}")

def main():
    """Main evaluation function with enhanced visualization"""
    print("Starting comprehensive model evaluation...")
    
    # Load test data
    X_test, y_test, feature_columns = load_test_data()
    
    # Get number of features
    num_features = get_model_input_features()
    
    # Check if FL model exists
    fl_model = load_model(FL_MODEL_PATH, num_features)
    
    # Check if centralized model exists, if not, train it
    centralized_model = load_model(CENTRALIZED_MODEL_PATH, num_features)
    if centralized_model is None:
        # For centralized model, we need to load and combine all client data
        print("Centralized model not found. Training a new centralized model...")
        
        # Combine data from all clients for centralized training
        client_dfs = []
        for client_id in [1, 2, 3]:  # Adjust based on your client count
            try:
                client_df = pd.read_csv(os.path.join(DATA_DIR, f"client_{client_id}_train.csv"))
                client_dfs.append(client_df)
                print(f"Loaded client_{client_id}_train.csv with {len(client_df)} samples")
            except FileNotFoundError:
                print(f"Client {client_id} train data not found, skipping")
        
        if client_dfs:
            # Combine all client data
            combined_df = pd.concat(client_dfs, ignore_index=True)
            print(f"Combined {len(combined_df)} samples for centralized training")
            
            # Extract features and labels
            X_train_centralized = combined_df.drop(columns=['FLAG']).values.astype(np.float32)
            y_train_centralized = combined_df['FLAG'].values.astype(np.float32)
            
            # Train centralized model
            centralized_model = train_centralized_model(X_train_centralized, y_train_centralized, num_features)
            
            # Save centralized model
            torch.save(centralized_model.state_dict(), CENTRALIZED_MODEL_PATH)
            print(f"Centralized model saved to {CENTRALIZED_MODEL_PATH}")
        else:
            print("No client data found for centralized training")
    
    # Evaluate models
    fl_metrics = evaluate_model(fl_model, X_test, y_test)
    centralized_metrics = evaluate_model(centralized_model, X_test, y_test)
    
    # Print results
    print("\nEvaluation Results:")
    print("------------------")
    print(f"FL Model AUC: {fl_metrics['auc']:.4f}, F1: {fl_metrics['f1']:.4f}")
    print(f"Centralized Model AUC: {centralized_metrics['auc']:.4f}, F1: {centralized_metrics['f1']:.4f}")
    
    # Generate enhanced visualizations
    plot_roc_curve(y_test, fl_metrics['probabilities'], 
                  centralized_metrics['probabilities'] if centralized_model else None)
    
    plot_precision_recall_curve(y_test, fl_metrics['probabilities'],
                               centralized_metrics['probabilities'] if centralized_model else None)
    
    plot_confusion_matrix(y_test, fl_metrics['predictions'],
                         centralized_metrics['predictions'] if centralized_model else None)
    
    # Sample privacy-utility tradeoff (simulated data)
    # In real implementation, you would run the FL system with different noise settings
    noise_levels = [0.001, 0.005, 0.01, 0.05, 0.1]
    utility_metrics = {
        'AUC': [0.98, 0.975, 0.97, 0.95, 0.92],
        'F1': [0.97, 0.96, 0.94, 0.91, 0.88]
    }
    plot_privacy_vs_utility(noise_levels, utility_metrics)
    
    # Sample client contributions (simulated data)
    # In real implementation, you would evaluate each client model separately
    client_metrics = {
        '1': {'auc': 0.95, 'f1': 0.92, 'precision': 0.94, 'recall': 0.90, 'accuracy': 0.97},
        '2': {'auc': 0.97, 'f1': 0.94, 'precision': 0.95, 'recall': 0.93, 'accuracy': 0.98},
        '3': {'auc': 0.93, 'f1': 0.91, 'precision': 0.92, 'recall': 0.89, 'accuracy': 0.96}
    }
    plot_client_contributions(client_metrics)
    
    # Generate traditional comparison plot
    plot_comparison(fl_metrics, centralized_metrics)
    
    # Save comprehensive metrics
    save_full_comparison(fl_metrics, centralized_metrics)
    
    print("Comprehensive evaluation complete!")

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

if __name__ == "__main__":
    main()