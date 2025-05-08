document.getElementById('transactionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = Object.fromEntries(new FormData(e.target));
    const resultsDiv = document.getElementById('results');
    const submitBtn = e.target.querySelector('button');
    
    // Remove FLAG from the data being sent
    delete formData.FLAG;
    
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(formData)
        });

        if (!response.ok) throw new Error('Analysis failed');
        
        const result = await response.json();
        const resultClass = result.result.toLowerCase();
        
        resultsDiv.classList.remove('hidden');
        resultsDiv.querySelector('.result-text').textContent = result.result;
        resultsDiv.querySelector('.result-text').className = `result-text ${resultClass}`;
        resultsDiv.querySelector('.confidence-value').textContent = result.confidence;
        
        const gauge = document.querySelector('.gauge-meter');
        gauge.style.transform = `rotate(${result.probability * 180 - 90}deg)`;
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing transaction. Please try again.');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Transaction';
    }
});