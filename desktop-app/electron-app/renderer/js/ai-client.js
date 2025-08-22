/**
 * AI Client for communicating with the Autonomous AI Agent API
 * Handles all API interactions with proper error handling and retry logic
 */

class AIClient {
    constructor() {
        this.endpoint = null;
        this.token = null;
        this.timeout = 30000; // 30 seconds
        this.retryAttempts = 3;
        this.retryDelay = 1000; // 1 second
        
        // Load configuration
        this.loadConfig();
        
        // Connection status
        this.isConnected = false;
        this.lastError = null;
        
        // Request queue for handling multiple requests
        this.requestQueue = [];
        this.isProcessingQueue = false;
    }

    /**
     * Load configuration from storage
     */
    loadConfig() {
        try {
            const config = configAPI.getAll();
            this.endpoint = config.apiEndpoint || 'https://your-deployment.vercel.app';
            this.token = config.apiToken || 'autonomous-ai-agent-2024';
            this.timeout = config.apiTimeout || 30000;
            
            logAPI.info('AI Client configuration loaded', { endpoint: this.endpoint });
        } catch (error) {
            logAPI.error('Failed to load AI client configuration', error);
            this.setDefaultConfig();
        }
    }

    /**
     * Set default configuration
     */
    setDefaultConfig() {
        this.endpoint = 'https://your-deployment.vercel.app';
        this.token = 'autonomous-ai-agent-2024';
        this.timeout = 30000;
        
        // Save default config
        configAPI.set('apiEndpoint', this.endpoint);
        configAPI.set('apiToken', this.token);
        configAPI.set('apiTimeout', this.timeout);
    }

    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        if (newConfig.endpoint) {
            this.endpoint = newConfig.endpoint;
            configAPI.set('apiEndpoint', this.endpoint);
        }
        
        if (newConfig.token) {
            this.token = newConfig.token;
            configAPI.set('apiToken', this.token);
        }
        
        if (newConfig.timeout) {
            this.timeout = newConfig.timeout;
            configAPI.set('apiTimeout', this.timeout);
        }
        
        logAPI.info('AI Client configuration updated');
    }

    /**
     * Get request headers
     */
    getHeaders() {
        return {
            'Authorization': `Bearer ${this.token}`,
            'Content-Type': 'application/json',
            'User-Agent': 'AutonomousAI-Desktop/1.0.0'
        };
    }

    /**
     * Make HTTP request with retry logic
     */
    async makeRequest(endpoint, options = {}) {
        const url = `${this.endpoint}${endpoint}`;
        const requestOptions = {
            ...options,
            headers: {
                ...this.getHeaders(),
                ...options.headers
            }
        };

        for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
            try {
                performanceAPI.mark(`request-start-${endpoint}`);
                
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.timeout);
                
                const response = await fetch(url, {
                    ...requestOptions,
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                performanceAPI.mark(`request-end-${endpoint}`);
                
                const duration = performanceAPI.measure(
                    `request-${endpoint}`,
                    `request-start-${endpoint}`,
                    `request-end-${endpoint}`
                );
                
                logAPI.debug(`Request to ${endpoint} took ${duration}ms`);

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new APIError(
                        `HTTP ${response.status}: ${response.statusText}`,
                        response.status,
                        errorData
                    );
                }

                const data = await response.json();
                this.isConnected = true;
                this.lastError = null;
                
                return data;

            } catch (error) {
                logAPI.warn(`Request attempt ${attempt} failed:`, error.message);
                
                if (attempt === this.retryAttempts) {
                    this.isConnected = false;
                    this.lastError = error;
                    throw error;
                }
                
                // Exponential backoff
                const delay = this.retryDelay * Math.pow(2, attempt - 1);
                await this.sleep(delay);
            }
        }
    }

    /**
     * Sleep utility for retry delays
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Test API connection
     */
    async testConnection() {
        try {
            const response = await this.makeRequest('/', { method: 'GET' });
            return {
                success: true,
                status: response.status,
                version: response.version,
                timestamp: response.timestamp
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                status: error.status
            };
        }
    }

    /**
     * Generate code
     */
    async generateCode(prompt, language = 'python', context = null) {
        try {
            const response = await this.makeRequest('/generate', {
                method: 'POST',
                body: JSON.stringify({
                    prompt,
                    language,
                    context
                })
            });

            logAPI.info('Code generation completed', {
                language,
                confidence: response.confidence,
                codeLength: response.code ? response.code.length : 0
            });

            return response;
        } catch (error) {
            logAPI.error('Code generation failed', error);
            throw error;
        }
    }

    /**
     * Search web for coding information
     */
    async searchWeb(query, depth = 5, includeCode = true) {
        try {
            const response = await this.makeRequest('/search', {
                method: 'POST',
                body: JSON.stringify({
                    query,
                    depth,
                    include_code: includeCode
                })
            });

            logAPI.info('Web search completed', {
                query,
                resultsCount: response.results ? response.results.length : 0,
                confidence: response.confidence
            });

            return response;
        } catch (error) {
            logAPI.error('Web search failed', error);
            throw error;
        }
    }

    /**
     * Reason about problem step by step
     */
    async reasonStepByStep(problem, domain = 'coding', includeMath = true) {
        try {
            const response = await this.makeRequest('/reason', {
                method: 'POST',
                body: JSON.stringify({
                    problem,
                    domain,
                    include_math: includeMath
                })
            });

            logAPI.info('Reasoning completed', {
                domain,
                stepsCount: response.reasoning_steps ? response.reasoning_steps.length : 0,
                confidence: response.confidence
            });

            return response;
        } catch (error) {
            logAPI.error('Reasoning failed', error);
            throw error;
        }
    }

    /**
     * Trigger agent training
     */
    async triggerTraining(datasetName = null, trainingType = 'rlhf', iterations = 10) {
        try {
            const response = await this.makeRequest('/train', {
                method: 'POST',
                body: JSON.stringify({
                    dataset_name: datasetName,
                    training_type: trainingType,
                    iterations
                })
            });

            logAPI.info('Training triggered', {
                trainingType,
                iterations,
                status: response.status
            });

            return response;
        } catch (error) {
            logAPI.error('Training trigger failed', error);
            throw error;
        }
    }

    /**
     * Get agent status
     */
    async getStatus() {
        try {
            const response = await this.makeRequest('/status', {
                method: 'GET'
            });

            logAPI.debug('Status retrieved', {
                modelLoaded: response.agent_status ? response.agent_status.model_loaded : false,
                memoryUsage: response.memory_usage ? response.memory_usage.memory_utilization : 0
            });

            return response;
        } catch (error) {
            logAPI.error('Status retrieval failed', error);
            throw error;
        }
    }

    /**
     * Provide feedback
     */
    async provideFeedback(rating, comments = null, specificFeedback = null) {
        try {
            const response = await this.makeRequest('/feedback', {
                method: 'POST',
                body: JSON.stringify({
                    rating,
                    comments,
                    specific_feedback: specificFeedback
                })
            });

            logAPI.info('Feedback submitted', { rating });

            return response;
        } catch (error) {
            logAPI.error('Feedback submission failed', error);
            throw error;
        }
    }

    /**
     * Stream response (for real-time updates)
     */
    async streamRequest(endpoint, options = {}, onChunk = null) {
        const url = `${this.endpoint}${endpoint}`;
        const requestOptions = {
            ...options,
            headers: {
                ...this.getHeaders(),
                ...options.headers
            }
        };

        try {
            const response = await fetch(url, requestOptions);
            
            if (!response.ok) {
                throw new APIError(`HTTP ${response.status}: ${response.statusText}`, response.status);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer
                
                for (const line of lines) {
                    if (line.trim() && onChunk) {
                        try {
                            const data = JSON.parse(line);
                            onChunk(data);
                        } catch (error) {
                            logAPI.warn('Failed to parse streaming data', error);
                        }
                    }
                }
            }

        } catch (error) {
            logAPI.error('Streaming request failed', error);
            throw error;
        }
    }

    /**
     * Get connection status
     */
    getConnectionStatus() {
        return {
            connected: this.isConnected,
            endpoint: this.endpoint,
            lastError: this.lastError ? this.lastError.message : null
        };
    }

    /**
     * Validate configuration
     */
    validateConfig() {
        const errors = [];

        if (!this.endpoint) {
            errors.push('API endpoint is required');
        } else if (!securityAPI.isValidURL(this.endpoint)) {
            errors.push('API endpoint must be a valid URL');
        }

        if (!this.token) {
            errors.push('API token is required');
        } else if (this.token.length < 10) {
            errors.push('API token appears to be invalid (too short)');
        }

        if (this.timeout < 5000) {
            errors.push('Timeout must be at least 5 seconds');
        }

        return {
            isValid: errors.length === 0,
            errors
        };
    }

    /**
     * Get usage statistics
     */
    getUsageStats() {
        const stats = configAPI.get('usageStats', {
            totalRequests: 0,
            successfulRequests: 0,
            failedRequests: 0,
            totalTokens: 0,
            averageResponseTime: 0,
            lastUsed: null
        });

        return stats;
    }

    /**
     * Update usage statistics
     */
    updateUsageStats(requestType, success, responseTime = 0, tokens = 0) {
        const stats = this.getUsageStats();
        
        stats.totalRequests++;
        if (success) {
            stats.successfulRequests++;
        } else {
            stats.failedRequests++;
        }
        
        stats.totalTokens += tokens;
        stats.averageResponseTime = ((stats.averageResponseTime * (stats.totalRequests - 1)) + responseTime) / stats.totalRequests;
        stats.lastUsed = new Date().toISOString();

        configAPI.set('usageStats', stats);
    }
}

/**
 * Custom API Error class
 */
class APIError extends Error {
    constructor(message, status = null, data = null) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.data = data;
    }
}

/**
 * Rate limiter for API requests
 */
class RateLimiter {
    constructor(maxRequests = 60, windowMs = 60000) {
        this.maxRequests = maxRequests;
        this.windowMs = windowMs;
        this.requests = [];
    }

    canMakeRequest() {
        const now = Date.now();
        const windowStart = now - this.windowMs;
        
        // Remove old requests
        this.requests = this.requests.filter(time => time > windowStart);
        
        return this.requests.length < this.maxRequests;
    }

    recordRequest() {
        this.requests.push(Date.now());
    }

    getTimeUntilReset() {
        if (this.requests.length === 0) return 0;
        
        const oldestRequest = Math.min(...this.requests);
        const resetTime = oldestRequest + this.windowMs;
        
        return Math.max(0, resetTime - Date.now());
    }
}

// Create global AI client instance
const aiClient = new AIClient();
const rateLimiter = new RateLimiter();

// Export for use in other modules
window.aiClient = aiClient;
window.rateLimiter = rateLimiter;
window.APIError = APIError;
