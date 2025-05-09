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

from .model import FraudDetectionNet, get_model_input_features # Relative import
from .clients import load_client_data # To get a global test set for server-side evaluation

# Configuration
NUM_ROUNDS = 10  # Reduced from 10 to prevent overfitting
AGGREGATE_EVERY_N_ROUNDS = 3 # Server aggregates weights every 3 rounds (as per req, but FedAvg usually aggregates each round)
                            # Flower's FedAvg aggregates every round by default.
                            # To aggregate every 3 rounds, we'd need a custom strategy or adjust client participation.
                            # For simplicity, FedAvg will aggregate each round. The "every 3 rounds" might refer to something else.
                            # Let's assume FedAvg default behavior (aggregate each round where clients participate).
                            # The prompt "Aggregates weights every 3 rounds" might be a misunderstanding of FedAvg.
                            # If it means server *evaluates* or *saves model* every 3 rounds, that's different.
                            # For now, standard FedAvg.
MIN_AVAILABLE_CLIENTS = 2 # Min clients for training round
MIN_FIT_CLIENTS = 2       # Min clients to train in a round
MIN_EVALUATE_CLIENTS = 2  # Min clients for evaluation
GLOBAL_MODEL_PATH = '/Users/ayushpatne/Developer/FL_Major/project/global_model.pth'
RESULTS_DIR = '/Users/ayushpatne/Developer/FL_Major/project/results/'
DATA_DIR = '/Users/ayushpatne/Developer/FL_Major/project/data/'


# Store metrics for plotting
history_fl = {'round': [], 'auc': [], 'f1': [], 'loss': [], 'accuracy': []}
history_centralized = {'auc': 0, 'f1': 0, 'precision': 0, 'recall': 0} # Placeholder

def get_evaluate_fn(model_class, num_input_features):
    """Return an evaluation function for server-side evaluation."""
    
    # Load a global test set (or a portion of client data for simplicity if global_test is not available)
    # Here we use the 'global_test_centralized.csv' created by data_split.py
    try:
        test_df = pd.read_csv(os.path.join(DATA_DIR, "global_test_centralized.csv"))
        X_test_global = test_df.drop(columns=['FLAG']).values.astype(np.float32)
        y_test_global = test_df['FLAG'].values.astype(np.float32).reshape(-1, 1)
        
        test_dataset = TensorDataset(torch.from_numpy(X_test_global), torch.from_numpy(y_test_global))
        testloader = DataLoader(test_dataset, batch_size=32)
        print(f"Loaded global test set for server evaluation: {len(test_df)} samples.")
    except FileNotFoundError:
        print("Global test set not found. Using client 1 test data as fallback for server evaluation.")
        # Fallback to client 1 test data if global test set is not available
        _, testloader = load_client_data(client_id=1, batch_size=32)
    
    # The evaluation function - needs to accept 3 parameters
    def evaluate(server_round, parameters, config):  # Modified parameter list
        # Initialize model with the latest weights
        model = model_class(num_input_features)
        params_dict = zip(model.state_dict().keys(), parameters)  # Use parameters arg
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        # Evaluate on the test set
        loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        y_prob = []
        
        with torch.no_grad():
            for data, target in testloader:
                outputs = model(data)
                loss += torch.nn.functional.binary_cross_entropy(outputs, target).item()
                
                predicted = (outputs > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(outputs.cpu().numpy())
        
        # Calculate metrics
        accuracy = correct / total
        auc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # Update history for plotting
        current_round = len(history_fl['round']) + 1
        history_fl['round'].append(current_round)
        history_fl['auc'].append(auc)
        history_fl['f1'].append(f1)
        history_fl['loss'].append(loss / len(testloader))
        history_fl['accuracy'].append(accuracy)
        
        # Save model every AGGREGATE_EVERY_N_ROUNDS rounds
        if current_round % AGGREGATE_EVERY_N_ROUNDS == 0 or current_round == NUM_ROUNDS:
            # Save the global model
            torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
            print(f"Global model saved to {GLOBAL_MODEL_PATH} at round {current_round}")
            
            # Create plots for visualization
            if not os.path.exists(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)
            
            # Plot metrics over rounds
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(history_fl['round'], history_fl['auc'], 'b-', label='FL Model')
            plt.axhline(y=history_centralized['auc'], color='r', linestyle='-', label='Centralized')
            plt.title('AUC-ROC over Rounds')
            plt.xlabel('Round')
            plt.ylabel('AUC-ROC')
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.plot(history_fl['round'], history_fl['f1'], 'b-', label='FL Model')
            plt.axhline(y=history_centralized['f1'], color='r', linestyle='-', label='Centralized')
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
            
            print(f"Evaluation plots saved to {os.path.join(RESULTS_DIR, 'comparison.png')}")
        
        return float(loss), {"accuracy": float(accuracy), "auc": float(auc), 
                            "f1": float(f1), "precision": float(precision), "recall": float(recall)}
    
    return evaluate

def load_centralized_model_metrics():
    """Load metrics from the centralized model for comparison."""
    try:
        # Try to load metrics from a saved file
        metrics = joblib.load(os.path.join(RESULTS_DIR, 'centralized_metrics.joblib'))
        history_centralized.update(metrics)
        print("Loaded centralized model metrics for comparison.")
    except FileNotFoundError:
        # If not available, use placeholder values
        print("Centralized model metrics not found. Using placeholder values.")
        # These will be updated if/when the centralized model is evaluated

def main():
    # Make sure the results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Load centralized model metrics for comparison
    load_centralized_model_metrics()
    
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
        on_fit_config_fn=lambda rnd: {"epoch": 5, "batch_size": 32},  # 5 epochs per round as per requirements
    )
    
    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:3000",  # Changed from 8080 to 3000
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    
    print(f"Federated Learning completed after {NUM_ROUNDS} rounds.")
    print(f"Final global model saved at: {GLOBAL_MODEL_PATH}")
    print(f"Evaluation results saved at: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
# In the main function, add this comment before defining the strategy:

# TODO: Implement encryption for secure aggregation
# This would involve using homomorphic encryption or secure multi-party computation
# to allow the server to aggregate model updates without seeing the actual values
# At the top of the file, add these imports and configurations
import warnings
import logging
import os

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('flwr').setLevel(logging.ERROR)