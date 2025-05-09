// FL Fraud Detection Dashboard JavaScript

// Global variables
let performanceChart = null;
let updateInterval = null;

// Initialize the dashboard when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the performance chart
    initializeChart();
    
    // Set up event listeners
    setupEventListeners();
    
    // Start periodic updates
    startPeriodicUpdates();
});

// Initialize the Plotly chart
function initializeChart() {
    // Fetch initial data and create chart
    fetch('/api/fl_metrics')
        .then(response => response.json())
        .then(data => {
            // Parse the Plotly JSON
            const chartData = JSON.parse(data.plot);
            
            // Create the chart
            Plotly.newPlot('performance-chart', chartData.data, chartData.layout);
            
            // Update global reference
            performanceChart = document.getElementById('performance-chart');
        })
        .catch(error => console.error('Error initializing chart:', error));
}

// Set up event listeners for interactive elements
function setupEventListeners() {
    // Prediction form submission
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(event) {
            event.preventDefault();
            submitPrediction();
        });
    }
    
    // Simulation control buttons
    const startSimulationBtn = document.getElementById('start-simulation');
    const stopSimulationBtn = document.getElementById('stop-simulation');
    
    if (startSimulationBtn) {
        startSimulationBtn.addEventListener('click', function() {
            startSimulation();
        });
    }
    
    if (stopSimulationBtn) {
        stopSimulationBtn.addEventListener('click', function() {
            stopSimulation();
        });
    }
}

// Start periodic updates for dashboard data
function startPeriodicUpdates() {
    // Clear any existing interval
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    // Update immediately
    updateClientStatus();
    updatePerformanceChart();
    updateRecentTransactions();
    
    // Set interval for updates (every 5 seconds)
    updateInterval = setInterval(function() {
        updateClientStatus();
        updatePerformanceChart();
        updateRecentTransactions();
    }, 5000);
}

// Update client status cards
function updateClientStatus() {
    fetch('/api/client_status')
        .then(response => response.json())
        .then(data => {
            // Update each client card
            for (const [clientId, status] of Object.entries(data)) {
                const cardElement = document.querySelector(`.client-card:nth-child(${status.id})`);
                if (cardElement) {
                    cardElement.querySelector('.fraud-ratio').textContent = `${(status.fraud_ratio * 100).toFixed(2)}%`;
                    cardElement.querySelector('.samples').textContent = status.samples;
                    cardElement.querySelector('.accuracy').textContent = `${(status.accuracy * 100).toFixed(2)}%`;
                    cardElement.querySelector('.last-update').textContent = status.last_update || 'Never';
                }
            }
        })
        .catch(error => console.error('Error updating client status:', error));
}

// Update the performance chart
function updatePerformanceChart() {
    fetch('/api/fl_metrics')
        .then(response => response.json())
        .then(data => {
            // Parse the Plotly JSON
            const chartData = JSON.parse(data.plot);
            
            // Update the chart
            Plotly.react('performance-chart', chartData.data, chartData.layout);
        })
        .catch(error => console.error('Error updating performance chart:', error));
}

// Update recent transactions table
function updateRecentTransactions() {
    fetch('/api/recent_transactions')
        .then(response => response.json())
        .then(transactions => {
            const tableBody = document.getElementById('recent-transactions');
            if (tableBody && transactions.length > 0) {
                // Clear existing rows
                tableBody.innerHTML = '';
                
                // Add new rows
                transactions.forEach(tx => {
                    const row = document.createElement('tr');
                    
                    // Add cells
                    row.innerHTML = `
                        <td>${tx.amount ? tx.amount.toFixed(2) : 'N/A'}</td>
                        <td>${tx.sender_id || 'N/A'}</td>
                        <td>${tx.receiver_id || 'N/A'}</td>
                        <td>${tx.timestamp || 'N/A'}</td>
                        <td class="${tx.FLAG ? 'fraud-flag-true' : 'fraud-flag-false'}">${tx.FLAG ? 'Yes' : 'No'}</td>
                    `;
                    
                    tableBody.appendChild(row);
                });
            } else if (tableBody && transactions.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="5" class="text-center">No transactions yet</td></tr>';
            }
        })
        .catch(error => console.error('Error updating recent transactions:', error));
}

// Submit prediction form
function submitPrediction() {
    const form = document.getElementById('prediction-form');
    const formData = new FormData(form);
    
    // Show loading state
    const submitButton = form.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    submitButton.textContent = 'Predicting...';
    submitButton.disabled = true;
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Reset button
        submitButton.textContent = originalText;
        submitButton.disabled = false;
        
        // Show prediction result
        const resultDiv = document.getElementById('prediction-result');
        const probabilitySpan = document.getElementById('fraud-probability');
        const classificationSpan = document.getElementById('fraud-classification');
        
        if (data.error) {
            // Show error
            resultDiv.style.display = 'block';
            resultDiv.querySelector('.alert').className = 'alert alert-danger';
            probabilitySpan.textContent = 'Error';
            classificationSpan.textContent = data.error;
        } else {
            // Show prediction
            resultDiv.style.display = 'block';
            
            // Format probability
            const probability = (data.fraud_probability * 100).toFixed(2) + '%';
            probabilitySpan.textContent = probability;
            
            // Set classification and alert style
            if (data.is_fraud) {
                resultDiv.querySelector('.alert').className = 'alert alert-danger';
                classificationSpan.textContent = 'FRAUD DETECTED';
            } else {
                resultDiv.querySelector('.alert').className = 'alert alert-success';
                classificationSpan.textContent = 'Legitimate Transaction';
            }
        }
    })
    .catch(error => {
        console.error('Error submitting prediction:', error);
        submitButton.textContent = originalText;
        submitButton.disabled = false;
        
        // Show error
        const resultDiv = document.getElementById('prediction-result');
        resultDiv.style.display = 'block';
        resultDiv.querySelector('.alert').className = 'alert alert-danger';
        document.getElementById('fraud-probability').textContent = 'Error';
        document.getElementById('fraud-classification').textContent = 'Network error. Please try again.';
    });
}

// Start transaction simulation
function startSimulation() {
    fetch('/simulate', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        console.log('Simulation status:', data.status);
        document.getElementById('simulation-status').textContent = 'Running';
        document.getElementById('simulation-status').className = 'running';
    })
    .catch(error => console.error('Error starting simulation:', error));
}

// Stop transaction simulation
function stopSimulation() {
    fetch('/stop_simulation', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        console.log('Simulation status:', data.status);
        document.getElementById('simulation-status').textContent = 'Stopped';
        document.getElementById('simulation-status').className = 'stopped';
    })
    .catch(error => console.error('Error stopping simulation:', error));
}