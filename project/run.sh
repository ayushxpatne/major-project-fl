#!/bin/bash

# Path to virtual environment
VENV_PATH="../venv"
PYTHON="$VENV_PATH/bin/python"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Create necessary directories
mkdir -p data
mkdir -p results

# Check if data exists, if not generate it
if [ ! -f "data/client_1_train.csv" ]; then
    echo "Generating data splits..."
    $PYTHON scripts/data_split.py
fi

# Generate synthetic transactions if they don't exist
if [ ! -f "data/simulated_transactions.csv" ]; then
    echo "Generating synthetic transactions..."
    $PYTHON scripts/simulate_tx.py
fi

# --- ADD THIS SECTION ---
# Train and evaluate the centralized model, and save its metrics
# This must run before the server starts so the server can load its metrics
echo "Training and evaluating centralized model (and saving metrics)..."
# Train and evaluating centralized model (and saving metrics)...
$PYTHON scripts/evaluate.py
# --- END ADDITION ---

# Start FL server in background
echo "Starting FL server..."
$PYTHON -m fl_system.server &
SERVER_PID=$!

# Wait for server to initialize
sleep 3

# Start FL clients in background
echo "Starting FL clients..."
$PYTHON -m fl_system.clients 1 &
CLIENT1_PID=$!
$PYTHON -m fl_system.clients 2 &
CLIENT2_PID=$!
$PYTHON -m fl_system.clients 3 &
CLIENT3_PID=$!

# Wait for clients to initialize
sleep 2

Launch Flask dashboard (uncomment if you want to use it)
echo "Starting Flask dashboard..."
$PYTHON app/app.py

# # Cleanup function
# cleanup() {
#     echo "Shutting down..."
#     kill $SERVER_PID $CLIENT1_PID $CLIENT2_PID $CLIENT3_PID
#     # Deactivate virtual environment
#     deactivate
#     exit 0
# }

# Set up trap for cleanup
# trap cleanup INT TERM

# Keep the script running until interrupted
wait