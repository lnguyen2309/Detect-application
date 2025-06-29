async function runDetection() {
            const chiffre1 = document.getElementById('chiffre1').value;
            const chiffre2 = document.getElementById('chiffre2').value;
            const resultContent = document.getElementById('resultContent');
            const detectBtn = document.querySelector('.detect-btn');
            
            // Disable button and show loading
            detectBtn.disabled = true;
            detectBtn.textContent = 'Detecting...';
            resultContent.innerHTML = '<p class="loading">Running detection...</p>';
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        chiffre1: chiffre1,
                        chiffre2: chiffre2
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const result = data.result;
                    resultContent.innerHTML = `
                        <div class="result-item">
                            <div class="result-label">Detection Result:</div>
                            <div class="result-value">${result.result}</div>
                        </div>
                        <div class="result-item">
                            <div class="result-label">Confidence:</div>
                            <div class="result-value">${(result.confidence * 100).toFixed(1)}%</div>
                        </div>
                        <div class="result-item">
                            <div class="result-label">Processed Values:</div>
                            <div class="result-value">${result.processed_values.join(', ')}</div>
                        </div>
                    `;
                } else {
                    resultContent.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                resultContent.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            } finally {
                // Re-enable button
                detectBtn.disabled = false;
                detectBtn.textContent = 'Detect';
            }
        }