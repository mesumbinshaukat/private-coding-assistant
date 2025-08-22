const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // API Health Check
  healthCheck: () => ipcRenderer.invoke('api-health-check'),
  
  // Code Generation
  generateCode: (prompt, language) => ipcRenderer.invoke('api-generate-code', { prompt, language }),
  
  // Web Search
  search: (query, depth) => ipcRenderer.invoke('api-search', { query, depth }),
  
  // Reasoning
  reason: (problem, domain, includeMath) => ipcRenderer.invoke('api-reason', { problem, domain, includeMath }),
  
  // API Status
  getStatus: () => ipcRenderer.invoke('api-status'),
  
  // File Operations
  saveFile: (content, filename) => ipcRenderer.invoke('save-file', { content, filename })
});
