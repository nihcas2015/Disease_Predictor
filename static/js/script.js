document.addEventListener('DOMContentLoaded', () => {
    const predictButton = document.getElementById('predict-button');
    const symptomsInput = document.getElementById('symptoms-input');
    const resultDisplay = document.getElementById('result-display');
    const spinner = document.getElementById('spinner');

    if (predictButton) {
        predictButton.addEventListener('click', async () => {
            const modelType = predictButton.dataset.model; // Get model type from button's data attribute
            const symptoms = symptomsInput.value.trim();

            if (!symptoms) {
                resultDisplay.innerHTML = `<p class="error">Please enter symptoms.</p>`;
                return;
            }

            // Clear previous results and show spinner
            resultDisplay.innerHTML = '';
            spinner.classList.remove('hidden');
            predictButton.disabled = true; // Disable button during request

            try {
                const response = await fetch(`/predict/${modelType}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ symptoms: symptoms }),
                });

                // Always hide spinner and re-enable button after fetch attempt
                spinner.classList.add('hidden');
                predictButton.disabled = false;

                const data = await response.json();

                if (!response.ok) {
                    // Handle errors from the backend (like validation errors)
                    resultDisplay.innerHTML = `<p class="error">Error: ${data.error || 'Prediction failed.'}</p>`;
                } else {
                    // Display successful prediction
                    let resultHTML = `
                        <p><strong>Result:</strong></p>
                        <p class="prediction">${data.disease}</p>
                        <p class="confidence">Confidence: ${data.confidence}%</p>
                    `;
                     // Display warning if present
                    if (data.warning) {
                        resultHTML += `<p class="warning">Warning: ${data.warning}</p>`;
                    }
                    resultDisplay.innerHTML = resultHTML;
                }

            } catch (error) {
                // Handle network errors or issues with the fetch itself
                console.error('Fetch Error:', error);
                spinner.classList.add('hidden');
                predictButton.disabled = false;
                resultDisplay.innerHTML = `<p class="error">An error occurred while contacting the server. Please try again.</p>`;
            }
        });
    }

    // Optional: Add hover effects dynamically if needed (though CSS handles the image hover)
    // const modelLinks = document.querySelectorAll('.model-link img');
    // modelLinks.forEach(img => {
    //     img.addEventListener('mouseover', () => { /* Add effects */ });
    //     img.addEventListener('mouseout', () => { /* Remove effects */ });
    // });
});