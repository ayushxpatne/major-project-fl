import os
import numpy as np
import pandas as pd
import time
import random
import joblib
from datetime import datetime, timedelta
from collections import defaultdict

# Add project root to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
DATA_DIR = '/Users/ayushpatne/Developer/FL_Major/project/data/'
OUTPUT_FILE = os.path.join(DATA_DIR, 'simulated_transactions.csv')
SCALER_PATH = os.path.join(DATA_DIR, 'scaler.joblib')
FEATURE_COLUMNS_PATH = os.path.join(DATA_DIR, 'feature_columns.joblib')

# Transaction patterns
FRAUD_RATIO = 0.10  # 10% fraudulent transactions
LARGE_AMOUNT_THRESHOLD = 100000  # Pattern 1: Large amounts (>$100k)
MICRO_TX_THRESHOLD = 10  # Pattern 2: <$10
MICRO_TX_COUNT_THRESHOLD = 10  # Pattern 2: 10+ transactions
MICRO_TX_TIME_WINDOW = 60  # Pattern 2: within 60 seconds

# User account tracking for micro-transaction pattern
user_tx_history = defaultdict(list)

def load_sample_data():
    """Load a sample of real data to understand the distribution"""
    try:
        # Try to load client_1 data as a reference
        sample_df = pd.read_csv(os.path.join(DATA_DIR, "client_1_train.csv"))
        print(f"Loaded sample data: {len(sample_df)} transactions")
        return sample_df
    except FileNotFoundError:
        print("No sample data found. Using default distributions.")
        return None

def load_scaler_and_features():
    """Load the scaler and feature columns used in training"""
    try:
        scaler = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        print("Loaded scaler and feature columns")
        return scaler, feature_columns
    except FileNotFoundError:
        print("Scaler or feature columns not found. Will create synthetic data without scaling.")
        return None, None

def generate_transaction(amount=None, sender_id=None, receiver_id=None, force_fraud=None):
    """Generate a single transaction with realistic features"""
    # Generate random sender and receiver if not provided
    if sender_id is None:
        sender_id = f"C{random.randint(1000, 9999)}"
    if receiver_id is None:
        receiver_id = f"M{random.randint(1000, 9999)}"
    
    # Generate random amount if not provided
    if amount is None:
        # Most transactions are small, but some are large
        if random.random() < 0.95:
            amount = random.uniform(10, 1000)
        else:
            amount = random.uniform(1000, 200000)
    
    # Determine if this should be a fraudulent transaction
    is_fraud = force_fraud
    if is_fraud is None:
        # Determine fraud based on patterns
        # Pattern 1: Large amount
        if amount > LARGE_AMOUNT_THRESHOLD:
            is_fraud = random.random() < 0.7  # 70% of large transactions are fraud
        else:
            # Base fraud rate for normal transactions
            is_fraud = random.random() < 0.05  # 5% base fraud rate
    
    # Generate balances based on transaction amount
    oldbalanceOrg = random.uniform(amount, amount * 10) if amount < 10000 else amount * 1.2
    newbalanceOrig = max(0, oldbalanceOrg - amount)
    
    oldbalanceDest = random.uniform(0, 50000)
    newbalanceDest = oldbalanceDest + amount
    
    # Generate transaction type
    tx_type = random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT'])
    if amount > 50000:
        tx_type = random.choice(['TRANSFER', 'CASH_OUT'])  # Large amounts are usually transfers or cash outs
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create transaction
    transaction = {
        'step': int(time.time() % 1000),  # Simplified step (time-based)
        'type': tx_type,
        'amount': amount,
        'nameOrig': sender_id,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'nameDest': receiver_id,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFlaggedFraud': 1 if amount > 200000 else 0,  # Flag very large transactions
        'FLAG': 1 if is_fraud else 0,
        'timestamp': timestamp  # Additional field for tracking
    }
    
    return transaction

def check_micro_tx_pattern(sender_id, amount, timestamp):
    """Check if this transaction is part of a micro-transaction fraud pattern"""
    if amount > MICRO_TX_THRESHOLD:
        return False
    
    # Add this transaction to the user's history
    tx_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    user_tx_history[sender_id].append((amount, tx_time))
    
    # Clean up old transactions (older than the time window)
    current_time = datetime.now()
    user_tx_history[sender_id] = [
        (amt, tm) for amt, tm in user_tx_history[sender_id]
        if (current_time - tm).total_seconds() <= MICRO_TX_TIME_WINDOW
    ]
    
    # Check if the pattern is detected
    if len(user_tx_history[sender_id]) >= MICRO_TX_COUNT_THRESHOLD:
        return True
    
    return False

def generate_batch_transactions(count, fraud_ratio=FRAUD_RATIO):
    """Generate a batch of transactions with the specified fraud ratio"""
    transactions = []
    
    # Calculate how many fraudulent transactions to generate
    fraud_count = int(count * fraud_ratio)
    normal_count = count - fraud_count
    
    # Generate normal transactions
    for _ in range(normal_count):
        tx = generate_transaction(force_fraud=False)
        transactions.append(tx)
    
    # Generate fraudulent transactions
    for _ in range(fraud_count):
        # Decide which fraud pattern to use
        if random.random() < 0.5:
            # Pattern 1: Large amount
            amount = random.uniform(LARGE_AMOUNT_THRESHOLD, LARGE_AMOUNT_THRESHOLD * 3)
            tx = generate_transaction(amount=amount, force_fraud=True)
        else:
            # Pattern 2: Micro-transactions
            # Generate a sender who will do multiple small transactions
            sender_id = f"C{random.randint(1000, 9999)}"
            amount = random.uniform(1, MICRO_TX_THRESHOLD)
            tx = generate_transaction(amount=amount, sender_id=sender_id, force_fraud=True)
            
            # Mark this as part of a micro-transaction pattern
            timestamp = tx['timestamp']
            check_micro_tx_pattern(sender_id, amount, timestamp)
        
        transactions.append(tx)
    
    # Shuffle to mix normal and fraudulent transactions
    random.shuffle(transactions)
    
    return transactions

def main(num_transactions=1000):
    """Generate synthetic transactions and save to CSV"""
    print(f"Generating {num_transactions} synthetic transactions...")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load sample data for reference
    sample_data = load_sample_data()
    
    # Load scaler and feature columns
    scaler, feature_columns = load_scaler_and_features()
    
    # Generate transactions
    transactions = generate_batch_transactions(num_transactions)
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} transactions to {OUTPUT_FILE}")
    
    # Print fraud statistics
    fraud_count = df['FLAG'].sum()
    print(f"Fraud ratio: {fraud_count / len(df):.2%} ({fraud_count} fraudulent transactions)")
    
    # Print pattern statistics
    large_amount_fraud = df[(df['FLAG'] == 1) & (df['amount'] > LARGE_AMOUNT_THRESHOLD)]
    print(f"Pattern 1 (Large Amount): {len(large_amount_fraud)} transactions")
    
    micro_tx_fraud = df[(df['FLAG'] == 1) & (df['amount'] <= MICRO_TX_THRESHOLD)]
    print(f"Pattern 2 (Micro-Transactions): {len(micro_tx_fraud)} transactions")

if __name__ == "__main__":
    main(1000)  # Generate 1000 transactions by default