/**
 * Preload script for Autonomous AI Agent Desktop App
 * Provides secure bridge between renderer and main process
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
    // App information
    getAppInfo: () => ipcRenderer.invoke('get-app-info'),
    
    // Window controls
    minimizeWindow: () => ipcRenderer.invoke('minimize-window'),
    maximizeWindow: () => ipcRenderer.invoke('maximize-window'),
    closeWindow: () => ipcRenderer.invoke('close-window'),
    
    // File operations
    showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),
    showOpenDialog: (options) => ipcRenderer.invoke('show-open-dialog', options),
    showMessageBox: (options) => ipcRenderer.invoke('show-message-box', options),
    
    // External links
    openExternal: (url) => ipcRenderer.invoke('open-external', url),
    
    // Updates
    checkForUpdates: () => ipcRenderer.invoke('check-for-updates'),
    
    // Menu handlers
    onMenuAction: (callback) => {
        ipcRenderer.on('menu-new-file', callback);
        ipcRenderer.on('menu-open-file', callback);
        ipcRenderer.on('menu-save-file', callback);
        ipcRenderer.on('menu-save-as', callback);
        ipcRenderer.on('menu-find', callback);
        ipcRenderer.on('menu-replace', callback);
    },
    
    // AI Agent handlers
    onAIAction: (callback) => {
        ipcRenderer.on('ai-open-assistant', callback);
        ipcRenderer.on('ai-generate-code', callback);
        ipcRenderer.on('ai-search-web', callback);
        ipcRenderer.on('ai-reason-problem', callback);
        ipcRenderer.on('ai-trigger-training', callback);
        ipcRenderer.on('ai-show-status', callback);
        ipcRenderer.on('ai-show-settings', callback);
    },
    
    // View handlers
    onViewAction: (callback) => {
        ipcRenderer.on('view-toggle-ai-panel', callback);
        ipcRenderer.on('view-toggle-explorer', callback);
    },
    
    // Help handlers
    onHelpAction: (callback) => {
        ipcRenderer.on('help-show-shortcuts', callback);
    },
    
    // App lifecycle
    onAppAction: (callback) => {
        ipcRenderer.on('app-before-quit', callback);
    },
    
    // Remove all listeners
    removeAllListeners: (channel) => {
        ipcRenderer.removeAllListeners(channel);
    }
});

// Expose a limited Node.js API for security
contextBridge.exposeInMainWorld('nodeAPI', {
    // Path utilities (safe subset)
    path: {
        join: (...args) => require('path').join(...args),
        dirname: (path) => require('path').dirname(path),
        basename: (path) => require('path').basename(path),
        extname: (path) => require('path').extname(path)
    },
    
    // Environment variables (safe subset)
    env: {
        NODE_ENV: process.env.NODE_ENV,
        PLATFORM: process.platform,
        ARCH: process.arch
    },
    
    // Version information
    versions: {
        node: process.versions.node,
        electron: process.versions.electron,
        chrome: process.versions.chrome
    }
});

// Configuration management
contextBridge.exposeInMainWorld('configAPI', {
    // Get configuration value
    get: (key, defaultValue = null) => {
        try {
            const config = localStorage.getItem('autonomousai-config');
            const parsedConfig = config ? JSON.parse(config) : {};
            return parsedConfig[key] !== undefined ? parsedConfig[key] : defaultValue;
        } catch (error) {
            console.error('Error getting config:', error);
            return defaultValue;
        }
    },
    
    // Set configuration value
    set: (key, value) => {
        try {
            const config = localStorage.getItem('autonomousai-config');
            const parsedConfig = config ? JSON.parse(config) : {};
            parsedConfig[key] = value;
            localStorage.setItem('autonomousai-config', JSON.stringify(parsedConfig));
            return true;
        } catch (error) {
            console.error('Error setting config:', error);
            return false;
        }
    },
    
    // Get all configuration
    getAll: () => {
        try {
            const config = localStorage.getItem('autonomousai-config');
            return config ? JSON.parse(config) : {};
        } catch (error) {
            console.error('Error getting all config:', error);
            return {};
        }
    },
    
    // Set multiple configuration values
    setAll: (configObject) => {
        try {
            localStorage.setItem('autonomousai-config', JSON.stringify(configObject));
            return true;
        } catch (error) {
            console.error('Error setting all config:', error);
            return false;
        }
    },
    
    // Reset configuration
    reset: () => {
        try {
            localStorage.removeItem('autonomousai-config');
            return true;
        } catch (error) {
            console.error('Error resetting config:', error);
            return false;
        }
    }
});

// Logging utilities
contextBridge.exposeInMainWorld('logAPI', {
    info: (message, ...args) => {
        console.info(`[INFO] ${message}`, ...args);
    },
    
    warn: (message, ...args) => {
        console.warn(`[WARN] ${message}`, ...args);
    },
    
    error: (message, ...args) => {
        console.error(`[ERROR] ${message}`, ...args);
    },
    
    debug: (message, ...args) => {
        if (process.env.NODE_ENV === 'development') {
            console.debug(`[DEBUG] ${message}`, ...args);
        }
    }
});

// Security utilities
contextBridge.exposeInMainWorld('securityAPI', {
    // Sanitize HTML to prevent XSS
    sanitizeHTML: (html) => {
        const div = document.createElement('div');
        div.textContent = html;
        return div.innerHTML;
    },
    
    // Validate URL
    isValidURL: (url) => {
        try {
            new URL(url);
            return true;
        } catch {
            return false;
        }
    },
    
    // Escape special characters
    escapeHTML: (text) => {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});

// Performance monitoring
contextBridge.exposeInMainWorld('performanceAPI', {
    // Mark performance measurement
    mark: (name) => {
        if (performance && performance.mark) {
            performance.mark(name);
        }
    },
    
    // Measure performance between marks
    measure: (name, startMark, endMark) => {
        if (performance && performance.measure) {
            try {
                performance.measure(name, startMark, endMark);
                const measure = performance.getEntriesByName(name)[0];
                return measure ? measure.duration : null;
            } catch (error) {
                console.warn('Performance measurement failed:', error);
                return null;
            }
        }
        return null;
    },
    
    // Get performance entries
    getEntries: (type = null) => {
        if (performance && performance.getEntries) {
            const entries = performance.getEntries();
            return type ? entries.filter(entry => entry.entryType === type) : entries;
        }
        return [];
    },
    
    // Clear performance marks and measures
    clear: () => {
        if (performance && performance.clearMarks && performance.clearMeasures) {
            performance.clearMarks();
            performance.clearMeasures();
        }
    }
});

// Theme and appearance
contextBridge.exposeInMainWorld('themeAPI', {
    // Get system theme preference
    getSystemTheme: () => {
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    },
    
    // Listen for theme changes
    onThemeChange: (callback) => {
        if (window.matchMedia) {
            const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
            const handler = (e) => callback(e.matches ? 'dark' : 'light');
            mediaQuery.addListener(handler);
            
            // Return cleanup function
            return () => mediaQuery.removeListener(handler);
        }
        return () => {};
    }
});

// Clipboard utilities
contextBridge.exposeInMainWorld('clipboardAPI', {
    // Write text to clipboard
    writeText: async (text) => {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (error) {
            console.error('Failed to write to clipboard:', error);
            return false;
        }
    },
    
    // Read text from clipboard
    readText: async () => {
        try {
            return await navigator.clipboard.readText();
        } catch (error) {
            console.error('Failed to read from clipboard:', error);
            return '';
        }
    }
});

// Notification utilities
contextBridge.exposeInMainWorld('notificationAPI', {
    // Show desktop notification
    show: (title, options = {}) => {
        if ('Notification' in window) {
            if (Notification.permission === 'granted') {
                return new Notification(title, options);
            } else if (Notification.permission === 'default') {
                Notification.requestPermission().then(permission => {
                    if (permission === 'granted') {
                        return new Notification(title, options);
                    }
                });
            }
        }
        return null;
    },
    
    // Request notification permission
    requestPermission: async () => {
        if ('Notification' in window) {
            return await Notification.requestPermission();
        }
        return 'denied';
    },
    
    // Check notification permission
    getPermission: () => {
        return 'Notification' in window ? Notification.permission : 'denied';
    }
});

// Development utilities (only in dev mode)
if (process.env.NODE_ENV === 'development') {
    contextBridge.exposeInMainWorld('devAPI', {
        // Reload the page
        reload: () => {
            window.location.reload();
        },
        
        // Open developer tools
        openDevTools: () => {
            ipcRenderer.send('open-dev-tools');
        },
        
        // Log debug information
        logDebug: (...args) => {
            console.log('[DEV]', ...args);
        }
    });
}

// Console override for better logging in development
if (process.env.NODE_ENV === 'development') {
    const originalConsole = window.console;
    window.console = {
        ...originalConsole,
        log: (...args) => originalConsole.log('[RENDERER]', ...args),
        info: (...args) => originalConsole.info('[RENDERER]', ...args),
        warn: (...args) => originalConsole.warn('[RENDERER]', ...args),
        error: (...args) => originalConsole.error('[RENDERER]', ...args)
    };
}
