import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
import os
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from .model import FraudDetectionNet, get_model_input_features  # Relative import

# Add this constant



# Configuration - Use relative paths for better portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
GLOBAL_MODEL_PATH = os.path.join(BASE_DIR, 'global_model.pth')
CENTRALIZED_MODEL_PATH = os.path.join(BASE_DIR, 'centralized_model.pth')

# FL parameters
NUM_ROUNDS = 10
AGGREGATE_EVERY_N_ROUNDS = 3  # This is for metrics evaluation and model saving, not aggregation of weights
MIN_AVAILABLE_CLIENTS = 2
MIN_FIT_CLIENTS = 2
MIN_EVALUATE_CLIENTS = 2

# Store metrics for plotting
history_fl = {'round': [], 'auc': [], 'f1': [], 'loss': [], 'accuracy': [], 'precision': [], 'recall': []}

def load_test_data():
    """Load the global test dataset for evaluation"""
    try:
        test_df = pd.read_csv(os.path.join(DATA_DIR, "global_test_centralized.csv"))
        X_test = test_df.drop(columns=['FLAG']).values.astype(np.float32)
        y_test = test_df['FLAG'].values.astype(np.float32).reshape(-1, 1)
        
        return X_test, y_test, torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
    except FileNotFoundError:
        print("Global test set not found. Using client 1 test data as fallback.")
        try:
            test_df = pd.read_csv(os.path.join(DATA_DIR, "client_1_test.csv"))
            X_test = test_df.drop(columns=['FLAG']).values.astype(np.float32)
            y_test = test_df['FLAG'].values.astype(np.float32).reshape(-1, 1)
            
            return X_test, y_test, torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
        except FileNotFoundError:
            raise FileNotFoundError("No test data found. Please run data_split.py first.")

def load_centralized_model_metrics():
    """Load metrics from the centralized model for comparison."""
    try:
        metrics_path = os.path.join(RESULTS_DIR, 'centralized_metrics.joblib')
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
            print("Loaded centralized model metrics for comparison:", metrics)
            return metrics
        else:
            print("Centralized model metrics not found.")
            
            # If centralized_model.pth exists but metrics don't, evaluate it
            if os.path.exists(CENTRALIZED_MODEL_PATH):
                print("Centralized model found. Evaluating to get metrics...")
                num_features = get_model_input_features()
                model = FraudDetectionNet(num_features)
                model.load_state_dict(torch.load(CENTRALIZED_MODEL_PATH))
                model.eval()
                
                # Load test data
                X_test, y_test, X_tensor, y_tensor = load_test_data()
                
                # Evaluate
                with torch.no_grad():
                    outputs = model(X_tensor)
                    loss = torch.nn.functional.binary_cross_entropy(outputs, y_tensor).item()
                    preds = (outputs > 0.5).float()
                    accuracy = ((preds == y_tensor).sum().item()) / len(y_tensor)
                    
                    # Convert to numpy for sklearn metrics
                    y_true = y_tensor.numpy().flatten()
                    y_pred = preds.numpy().flatten()
                    y_prob = outputs.numpy().flatten()
                    
                    auc = roc_auc_score(y_true, y_prob)
                    f1 = f1_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred)
                    recall = recall_score(y_true, y_pred)
                    
                    metrics = {
                        'auc': float(auc),
                        'f1': float(f1),
                        'precision': float(precision),
                        'recall': float(recall),
                        'accuracy': float(accuracy),
                        'loss': float(loss)
                    }
                    
                    # Save metrics
                    if not os.path.exists(RESULTS_DIR):
                        os.makedirs(RESULTS_DIR)
                    joblib.dump(metrics, metrics_path)
                    print(f"Centralized model evaluated and metrics saved: {metrics}")
                    return metrics
    except Exception as e:
        print(f"Error loading centralized metrics: {e}")
    
    # Return default values if no metrics found
    return {
        'auc': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'accuracy': 0.0,
        'loss': 0.0
    }

def get_evaluate_fn(model_class, num_input_features):
    """Return an evaluation function for server-side evaluation."""
    
    # Load test data upfront to avoid reloading during each evaluation
    X_test, y_test, X_tensor, y_tensor = load_test_data()
    print(f"Loaded test data for server evaluation: {len(X_test)} samples")
    
    # Load centralized model metrics for comparison
    centralized_metrics = load_centralized_model_metrics()
    
    # The evaluation function
    def evaluate(server_round, parameters, config):
        # Initialize model with the latest weights
        model = model_class(num_input_features)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        # Evaluate on the test set
        with torch.no_grad():
            outputs = model(X_tensor)
            loss = torch.nn.functional.binary_cross_entropy(outputs, y_tensor).item()
            preds = (outputs > 0.5).float()
            accuracy = ((preds == y_tensor).sum().item()) / len(y_tensor)
            
            # Convert to numpy for sklearn metrics
            y_true = y_tensor.numpy().flatten()
            y_pred = preds.numpy().flatten()
            y_prob = outputs.numpy().flatten()
            
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = 0.5
                print("Warning: AUC could not be computed due to single class or constant predictions.")
            
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
        
        # Update history for plotting
        history_fl['round'].append(server_round)
        history_fl['auc'].append(auc)
        history_fl['f1'].append(f1)
        history_fl['loss'].append(loss)
        history_fl['accuracy'].append(accuracy)
        history_fl['precision'].append(precision)
        history_fl['recall'].append(recall)
        
        print(f"\nRound {server_round} Evaluation Results:")
        print(f"FL Model - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
        print(f"Centralized - AUC: {centralized_metrics['auc']:.4f}, F1: {centralized_metrics['f1']:.4f}")
        
        # Save model periodically or at the end
        if server_round % AGGREGATE_EVERY_N_ROUNDS == 0 or server_round == NUM_ROUNDS:
            # Save the global model
            torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
            print(f"Global model saved to {GLOBAL_MODEL_PATH} at round {server_round}")
            
            # Create plots for visualization
            if not os.path.exists(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)
            
            # Plot metrics over rounds
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(history_fl['round'], history_fl['auc'], 'b-', label='FL Model')
            plt.axhline(y=centralized_metrics['auc'], color='r', linestyle='-', label='Centralized')
            plt.title('AUC-ROC over Rounds')
            plt.xlabel('Round')
            plt.ylabel('AUC-ROC')
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.plot(history_fl['round'], history_fl['f1'], 'b-', label='FL Model')
            plt.axhline(y=centralized_metrics['f1'], color='r', linestyle='-', label='Centralized')
            plt.title('F1 Score over Rounds')
            plt.xlabel('Round')
            plt.ylabel('F1 Score')
            plt.legend()
            
            plt.subplot(2, 2, 3)
            plt.plot(history_fl['round'], history_fl['loss'], 'g-')
            plt.title('Loss over Rounds')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            
            plt.subplot(2, 2, 4)
            plt.plot(history_fl['round'], history_fl['accuracy'], 'g-')
            plt.title('Accuracy over Rounds')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'comparison.png'))
            plt.close()
            
            # Save metrics for external use
            metrics_data = {
                'round': history_fl['round'],
                'auc': history_fl['auc'],
                'f1': history_fl['f1'],
                'loss': history_fl['loss'],
                'accuracy': history_fl['accuracy'],
                'precision': history_fl['precision'],
                'recall': history_fl['recall']
            }
            
            # Save as JSON for web access
            import json
            with open(os.path.join(RESULTS_DIR, 'fl_metrics.json'), 'w') as f:
                json.dump(metrics_data, f)
            
            # Save last round metrics in joblib format
            last_metrics = {
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'loss': loss
            }
            joblib.dump(last_metrics, os.path.join(RESULTS_DIR, 'fl_metrics.joblib'))
            
            print(f"Evaluation plots and metrics saved at {RESULTS_DIR}")
            
        return float(loss), {
            "accuracy": float(accuracy),
            "auc": float(auc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        }
    
    return evaluate

def plot_global_metrics(history_fl, results_dir):
    """
    Plots global FL metrics (AUC, F1, Loss, Accuracy, Precision, Recall) over rounds.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    metrics = ['auc', 'f1', 'loss', 'accuracy', 'precision', 'recall']
    titles = {
        'auc': 'AUC-ROC over Rounds (Global Model)',
        'f1': 'F1 Score over Rounds (Global Model)',
        'loss': 'Loss over Rounds (Global Model)',
        'accuracy': 'Accuracy over Rounds (Global Model)',
        'precision': 'Precision over Rounds (Global Model)',
        'recall': 'Recall over Rounds (Global Model)'
    }
    filenames = {
        'auc': 'global_auc_roc_over_rounds.png',
        'f1': 'global_f1_score_over_rounds.png',
        'loss': 'global_loss_over_rounds.png',
        'accuracy': 'global_accuracy_over_rounds.png',
        'precision': 'global_precision_over_rounds.png',
        'recall': 'global_recall_over_rounds.png'
    }

    # Ensure 'round' key exists before proceeding
    if 'round' not in history_fl or not history_fl['round']:
        print("No rounds data available for plotting global metrics.")
        return

    rounds = history_fl['round']

    for metric in metrics:
        if metric in history_fl:
            plt.figure(figsize=(10, 6))
            plt.plot(rounds, history_fl[metric], marker='o', linestyle='-', color='skyblue')
            plt.title(titles[metric])
            plt.xlabel('Federated Learning Round')
            plt.ylabel(metric.capitalize())
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, filenames[metric]))
            plt.close()
            print(f"Saved plot: {os.path.join(results_dir, filenames[metric])}")
        else:
            print(f"Metric '{metric}' not found in history_fl. Skipping plot for this metric.")

def main():
    # Make sure the results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Get the number of input features for the model
    num_features = get_model_input_features()
    
    # Define the Flower strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.8,  # Sample 80% of available clients for training
        fraction_evaluate=0.8,  # Sample 80% of available clients for evaluation
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        evaluate_fn=get_evaluate_fn(FraudDetectionNet, num_features),
        on_evaluate_config_fn=lambda rnd: {"round_num": rnd},  # ← this was missing
        on_fit_config_fn=lambda rnd: {
        "epoch": 5,
        "batch_size": 32,
        "round_num": rnd  
    },
    )
    
    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:3000",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    print("Federated Learning training complete. Plotting global metrics...")
    plot_global_metrics(history_fl, RESULTS_DIR)
    print(f"Federated Learning completed after {NUM_ROUNDS} rounds.")
    print(f"Final global model saved at: {GLOBAL_MODEL_PATH}")
    print(f"Evaluation results saved at: {RESULTS_DIR}")

if __name__ == "__main__":
    main()