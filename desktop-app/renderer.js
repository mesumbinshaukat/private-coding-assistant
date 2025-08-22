// Tab Navigation
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tabs
    initializeTabs();
    
    // Check API health on startup
    checkAPIHealth();
    
    // Set up event listeners
    setupEventListeners();
});

function initializeTabs() {
    const navBtns = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Update active nav button
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Show target tab
            tabContents.forEach(tab => {
                tab.classList.remove('active');
                if (tab.id === `${targetTab}-tab`) {
                    tab.classList.add('active');
                }
            });
        });
    });
}

function setupEventListeners() {
    // Add enter key support for inputs
    document.getElementById('codePrompt').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            generateCode();
        }
    });
    
    document.getElementById('searchQuery').addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    
    document.getElementById('reasonProblem').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            performReasoning();
        }
    });
}

// API Health Check
async function checkAPIHealth() {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusDot = statusIndicator.querySelector('.status-dot');
    const statusText = statusIndicator.querySelector('.status-text');
    
    try {
        statusText.textContent = 'Checking...';
        statusDot.className = 'status-dot';
        
        const result = await window.electronAPI.healthCheck();
        
        if (result.success) {
            statusDot.classList.add('online');
            statusText.textContent = 'Online';
        } else {
            statusDot.classList.add('offline');
            statusText.textContent = 'Offline';
        }
    } catch (error) {
        statusDot.classList.add('offline');
        statusText.textContent = 'Error';
        console.error('Health check failed:', error);
    }
}

// Code Generation
async function generateCode() {
    const prompt = document.getElementById('codePrompt').value.trim();
    const language = document.getElementById('codeLanguage').value;
    const outputSection = document.getElementById('codeOutput');
    
    if (!prompt) {
        showMessage('Please enter a code generation prompt.', 'error');
        return;
    }
    
    // Show loading state
    outputSection.innerHTML = `
        <h3>Generated Code</h3>
        <div class="output-placeholder">
            <div class="loading"></div>
            <p>Generating code...</p>
        </div>
    `;
    
    try {
        const result = await window.electronAPI.generateCode(prompt, language);
        
        if (result.success) {
            displayCodeOutput(result.data);
        } else {
            showMessage(`Code generation failed: ${result.error}`, 'error');
            outputSection.innerHTML = `
                <h3>Generated Code</h3>
                <div class="output-placeholder">Code generation failed. Please try again.</div>
            `;
        }
    } catch (error) {
        showMessage(`Error: ${error.message}`, 'error');
        outputSection.innerHTML = `
            <h3>Generated Code</h3>
            <div class="output-placeholder">An error occurred. Please try again.</div>
        `;
    }
}

function displayCodeOutput(data) {
    const outputSection = document.getElementById('codeOutput');
    
    const codeBlock = `
        <h3>Generated Code</h3>
        <div class="code-block">${escapeHtml(data.code)}</div>
        <div class="code-actions">
            <button class="btn btn-success" onclick="saveCode('${escapeHtml(data.code)}', '${data.language || 'python'}')">
                💾 Save Code
            </button>
        </div>
        <div class="message info">
            <strong>Method:</strong> ${data.method}<br>
            <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%<br>
            ${data.complexity ? `<strong>Complexity:</strong> ${data.complexity}<br>` : ''}
            <strong>Explanation:</strong> ${data.explanation}
        </div>
    `;
    
    outputSection.innerHTML = codeBlock;
}

// Web Search
async function performSearch() {
    const query = document.getElementById('searchQuery').value.trim();
    const depth = parseInt(document.getElementById('searchDepth').value);
    const outputSection = document.getElementById('searchOutput');
    
    if (!query) {
        showMessage('Please enter a search query.', 'error');
        return;
    }
    
    // Show loading state
    outputSection.innerHTML = `
        <h3>Search Results</h3>
        <div class="output-placeholder">
            <div class="loading"></div>
            <p>Searching...</p>
        </div>
    `;
    
    try {
        const result = await window.electronAPI.search(query, depth);
        
        if (result.success) {
            displaySearchOutput(result.data);
        } else {
            showMessage(`Search failed: ${result.error}`, 'error');
            outputSection.innerHTML = `
                <h3>Search Results</h3>
                <div class="output-placeholder">Search failed. Please try again.</div>
            `;
        }
    } catch (error) {
        showMessage(`Error: ${error.message}`, 'error');
        outputSection.innerHTML = `
            <h3>Search Results</h3>
            <div class="output-placeholder">An error occurred. Please try again.</div>
        `;
    }
}

function displaySearchOutput(data) {
    const outputSection = document.getElementById('searchOutput');
    
    let resultsHtml = `
        <h3>Search Results</h3>
        <div class="message success">
            <strong>Query:</strong> ${escapeHtml(data.query)}<br>
            <strong>Results Found:</strong> ${data.result_count}<br>
            <strong>Method:</strong> ${data.method}<br>
            <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%
        </div>
        <div class="message info">
            <strong>Synthesized Answer:</strong><br>
            ${escapeHtml(data.synthesized_answer)}
        </div>
    `;
    
    if (data.results && data.results.length > 0) {
        data.results.forEach((result, index) => {
            resultsHtml += `
                <div class="search-result">
                    <h4>${escapeHtml(result.title)}</h4>
                    <a href="${result.url}" class="url" target="_blank">${result.url}</a>
                    <div class="snippet">${escapeHtml(result.snippet)}</div>
                    <small>Type: ${result.type || 'unknown'}</small>
                </div>
            `;
        });
    } else {
        resultsHtml += '<div class="output-placeholder">No results found.</div>';
    }
    
    outputSection.innerHTML = resultsHtml;
}

// Reasoning
async function performReasoning() {
    const problem = document.getElementById('reasonProblem').value.trim();
    const domain = document.getElementById('reasonDomain').value;
    const includeMath = document.getElementById('includeMath').checked;
    const outputSection = document.getElementById('reasonOutput');
    
    if (!problem) {
        showMessage('Please enter a problem to reason about.', 'error');
        return;
    }
    
    // Show loading state
    outputSection.innerHTML = `
        <h3>Reasoning Analysis</h3>
        <div class="output-placeholder">
            <div class="loading"></div>
            <p>Analyzing problem...</p>
        </div>
    `;
    
    try {
        const result = await window.electronAPI.reason(problem, domain, includeMath);
        
        if (result.success) {
            displayReasoningOutput(result.data);
        } else {
            showMessage(`Reasoning failed: ${result.error}`, 'error');
            outputSection.innerHTML = `
                <h3>Reasoning Analysis</h3>
                <div class="output-placeholder">Reasoning failed. Please try again.</div>
            `;
        }
    } catch (error) {
        showMessage(`Error: ${error.message}`, 'error');
        outputSection.innerHTML = `
            <h3>Reasoning Analysis</h3>
            <div class="output-placeholder">An error occurred. Please try again.</div>
        `;
    }
}

function displayReasoningOutput(data) {
    const outputSection = document.getElementById('reasonOutput');
    
    let reasoningHtml = `
        <h3>Reasoning Analysis</h3>
        <div class="message info">
            <strong>Problem:</strong> ${escapeHtml(data.problem || 'N/A')}<br>
            <strong>Domain:</strong> ${data.domain || 'N/A'}<br>
            <strong>Method:</strong> ${data.method}<br>
            <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%
        </div>
    `;
    
    if (data.reasoning_steps && data.reasoning_steps.length > 0) {
        data.reasoning_steps.forEach(step => {
            reasoningHtml += `
                <div class="reasoning-step">
                    <div class="step-number">${step.step}</div>
                    <div class="step-title">${escapeHtml(step.description)}</div>
                    <div class="step-content">${escapeHtml(step.content)}</div>
                </div>
            `;
        });
    }
    
    if (data.solution) {
        reasoningHtml += `
            <div class="message success">
                <strong>Solution:</strong><br>
                ${escapeHtml(data.solution)}
            </div>
        `;
    }
    
    outputSection.innerHTML = reasoningHtml;
}

// API Status
async function checkStatus() {
    const statusDetails = document.getElementById('statusDetails');
    
    // Show loading state
    statusDetails.innerHTML = `
        <div class="output-placeholder">
            <div class="loading"></div>
            <p>Checking API status...</p>
        </div>
    `;
    
    try {
        const result = await window.electronAPI.getStatus();
        
        if (result.success) {
            displayStatusOutput(result.data);
        } else {
            showMessage(`Status check failed: ${result.error}`, 'error');
            statusDetails.innerHTML = `
                <div class="output-placeholder">Status check failed. Please try again.</div>
            `;
        }
    } catch (error) {
        showMessage(`Error: ${error.message}`, 'error');
        statusDetails.innerHTML = `
            <div class="output-placeholder">An error occurred. Please try again.</div>
        `;
    }
}

function displayStatusOutput(data) {
    const statusDetails = document.getElementById('statusDetails');
    
    const statusHtml = `
        <div class="message success">
            <strong>API Status:</strong> ${data.api_status}<br>
            <strong>Deployment Mode:</strong> ${data.deployment_mode || 'N/A'}<br>
            <strong>Available Features:</strong> ${data.available_features.join(', ')}<br>
            <strong>Timestamp:</strong> ${new Date(data.timestamp).toLocaleString()}
        </div>
    `;
    
    statusDetails.innerHTML = statusHtml;
}

// File Operations
async function saveCode(code, language) {
    try {
        const filename = `generated_code.${getFileExtension(language)}`;
        const result = await window.electronAPI.saveFile(code, filename);
        
        if (result.success) {
            showMessage(`Code saved successfully to: ${result.filePath}`, 'success');
        } else {
            showMessage(`Failed to save code: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error saving code: ${error.message}`, 'error');
    }
}

function getFileExtension(language) {
    const extensions = {
        'python': 'py',
        'javascript': 'js',
        'java': 'java',
        'cpp': 'cpp'
    };
    return extensions[language] || 'txt';
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showMessage(message, type = 'info') {
    // Remove existing messages
    const existingMessages = document.querySelectorAll('.message');
    existingMessages.forEach(msg => msg.remove());
    
    // Create new message
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = message;
    
    // Insert at the top of the main content
    const mainContent = document.querySelector('.main-content');
    mainContent.insertBefore(messageDiv, mainContent.firstChild);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.remove();
        }
    }, 5000);
}

// Global error handler
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    showMessage(`An unexpected error occurred: ${e.error.message}`, 'error');
});

// Unhandled promise rejection handler
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    showMessage(`An unexpected error occurred: ${e.reason}`, 'error');
});
