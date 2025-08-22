const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const axios = require('axios');

// Keep a global reference of the window object
let mainWindow;

// API Configuration
const API_BASE_URL = 'https://private-coding-assistant.vercel.app';
const API_TOKEN = 'autonomous-ai-agent-secret-key-2024'; // In production, use proper auth

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'assets/icon.png'),
    title: 'Autonomous AI Agent'
  });

  // Load the index.html file
  mainWindow.loadFile('index.html');

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }

  // Emitted when the window is closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// This method will be called when Electron has finished initialization
app.whenReady().then(createWindow);

// Quit when all windows are closed
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC Handlers for API communication
ipcMain.handle('api-health-check', async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('api-generate-code', async (event, { prompt, language }) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/generate`, {
      prompt,
      language
    }, {
      headers: {
        'Authorization': `Bearer ${API_TOKEN}`,
        'Content-Type': 'application/json'
      }
    });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('api-search', async (event, { query, depth }) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/search`, {
      query,
      depth
    }, {
      headers: {
        'Authorization': `Bearer ${API_TOKEN}`,
        'Content-Type': 'application/json'
      }
    });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('api-reason', async (event, { problem, domain, includeMath }) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/reason`, {
      problem,
      domain,
      include_math: includeMath
    }, {
      headers: {
        'Authorization': `Bearer ${API_TOKEN}`,
        'Content-Type': 'application/json'
      }
    });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('api-status', async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/status`, {
      headers: {
        'Authorization': `Bearer ${API_TOKEN}`
      }
    });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// File operations
ipcMain.handle('save-file', async (event, { content, filename }) => {
  try {
    const result = await dialog.showSaveDialog(mainWindow, {
      defaultPath: filename || 'generated_code.py',
      filters: [
        { name: 'Python Files', extensions: ['py'] },
        { name: 'JavaScript Files', extensions: ['js'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    });
    
    if (!result.canceled) {
      const fs = require('fs');
      fs.writeFileSync(result.filePath, content);
      return { success: true, filePath: result.filePath };
    }
    return { success: false, error: 'Save cancelled' };
  } catch (error) {
    return { success: false, error: error.message };
  }
});
