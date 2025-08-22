/**
 * Main Electron process for Autonomous AI Agent Desktop App
 * Handles application lifecycle, window management, and security
 */

const { app, BrowserWindow, Menu, dialog, shell, ipcMain } = require('electron');
const path = require('path');
const isDev = process.env.NODE_ENV === 'development';

// Keep a global reference of the window object
let mainWindow;
let splashWindow;

/**
 * Create the main application window
 */
function createMainWindow() {
    // Create the browser window
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1200,
        minHeight: 700,
        icon: path.join(__dirname, 'renderer/assets/icon.png'),
        show: false, // Don't show until ready
        titleBarStyle: 'default',
        webPreferences: {
            nodeIntegration: false, // Security: disable node integration
            contextIsolation: true, // Security: enable context isolation
            enableRemoteModule: false, // Security: disable remote module
            preload: path.join(__dirname, 'preload.js'),
            webSecurity: true,
            allowRunningInsecureContent: false
        }
    });

    // Load the app
    mainWindow.loadFile('renderer/index.html');

    // Show window when ready to prevent visual flash
    mainWindow.once('ready-to-show', () => {
        if (splashWindow) {
            splashWindow.close();
        }
        mainWindow.show();
        
        // Focus the window
        if (isDev) {
            mainWindow.webContents.openDevTools();
        }
    });

    // Handle window closed
    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // Handle external links
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });

    // Prevent navigation to external sites
    mainWindow.webContents.on('will-navigate', (event, navigationUrl) => {
        const parsedUrl = new URL(navigationUrl);
        
        if (parsedUrl.origin !== 'file://') {
            event.preventDefault();
            shell.openExternal(navigationUrl);
        }
    });

    return mainWindow;
}

/**
 * Create splash screen
 */
function createSplashWindow() {
    splashWindow = new BrowserWindow({
        width: 400,
        height: 300,
        frame: false,
        alwaysOnTop: true,
        transparent: true,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true
        }
    });

    splashWindow.loadFile('renderer/splash.html');

    splashWindow.on('closed', () => {
        splashWindow = null;
    });

    return splashWindow;
}

/**
 * Create application menu
 */
function createMenu() {
    const template = [
        {
            label: 'File',
            submenu: [
                {
                    label: 'New File',
                    accelerator: 'CmdOrCtrl+N',
                    click: () => {
                        mainWindow.webContents.send('menu-new-file');
                    }
                },
                {
                    label: 'Open File',
                    accelerator: 'CmdOrCtrl+O',
                    click: async () => {
                        const result = await dialog.showOpenDialog(mainWindow, {
                            properties: ['openFile'],
                            filters: [
                                { name: 'All Files', extensions: ['*'] },
                                { name: 'Python', extensions: ['py'] },
                                { name: 'JavaScript', extensions: ['js', 'ts'] },
                                { name: 'Text', extensions: ['txt', 'md'] }
                            ]
                        });

                        if (!result.canceled) {
                            mainWindow.webContents.send('menu-open-file', result.filePaths[0]);
                        }
                    }
                },
                {
                    label: 'Save',
                    accelerator: 'CmdOrCtrl+S',
                    click: () => {
                        mainWindow.webContents.send('menu-save-file');
                    }
                },
                {
                    label: 'Save As',
                    accelerator: 'CmdOrCtrl+Shift+S',
                    click: () => {
                        mainWindow.webContents.send('menu-save-as');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Exit',
                    accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
                    click: () => {
                        app.quit();
                    }
                }
            ]
        },
        {
            label: 'Edit',
            submenu: [
                { role: 'undo' },
                { role: 'redo' },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' },
                { role: 'selectall' },
                { type: 'separator' },
                {
                    label: 'Find',
                    accelerator: 'CmdOrCtrl+F',
                    click: () => {
                        mainWindow.webContents.send('menu-find');
                    }
                },
                {
                    label: 'Replace',
                    accelerator: 'CmdOrCtrl+H',
                    click: () => {
                        mainWindow.webContents.send('menu-replace');
                    }
                }
            ]
        },
        {
            label: 'AI Agent',
            submenu: [
                {
                    label: 'Open AI Assistant',
                    accelerator: 'CmdOrCtrl+Shift+A',
                    click: () => {
                        mainWindow.webContents.send('ai-open-assistant');
                    }
                },
                {
                    label: 'Generate Code',
                    accelerator: 'CmdOrCtrl+Shift+G',
                    click: () => {
                        mainWindow.webContents.send('ai-generate-code');
                    }
                },
                {
                    label: 'Search Web',
                    accelerator: 'CmdOrCtrl+Shift+S',
                    click: () => {
                        mainWindow.webContents.send('ai-search-web');
                    }
                },
                {
                    label: 'Reason About Problem',
                    accelerator: 'CmdOrCtrl+Shift+R',
                    click: () => {
                        mainWindow.webContents.send('ai-reason-problem');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Trigger Training',
                    accelerator: 'CmdOrCtrl+Shift+T',
                    click: () => {
                        mainWindow.webContents.send('ai-trigger-training');
                    }
                },
                {
                    label: 'Agent Status',
                    click: () => {
                        mainWindow.webContents.send('ai-show-status');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Settings',
                    click: () => {
                        mainWindow.webContents.send('ai-show-settings');
                    }
                }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' },
                { type: 'separator' },
                {
                    label: 'Toggle AI Panel',
                    accelerator: 'CmdOrCtrl+\\',
                    click: () => {
                        mainWindow.webContents.send('view-toggle-ai-panel');
                    }
                },
                {
                    label: 'Toggle File Explorer',
                    accelerator: 'CmdOrCtrl+Shift+E',
                    click: () => {
                        mainWindow.webContents.send('view-toggle-explorer');
                    }
                }
            ]
        },
        {
            label: 'Help',
            submenu: [
                {
                    label: 'Documentation',
                    click: () => {
                        shell.openExternal('https://github.com/your-repo/autonomous-ai-agent');
                    }
                },
                {
                    label: 'API Reference',
                    click: () => {
                        shell.openExternal('https://your-deployment.vercel.app/docs');
                    }
                },
                {
                    label: 'Keyboard Shortcuts',
                    click: () => {
                        mainWindow.webContents.send('help-show-shortcuts');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Report Issue',
                    click: () => {
                        shell.openExternal('https://github.com/your-repo/autonomous-ai-agent/issues');
                    }
                },
                {
                    label: 'About',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'About Autonomous AI Agent',
                            message: 'Autonomous AI Agent Desktop',
                            detail: 'Version 1.0.0\n\nAn AI-powered coding assistant that helps you write, debug, and optimize code.\n\n© 2024 Autonomous AI Agent Team'
                        });
                    }
                }
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
}

/**
 * Setup IPC handlers
 */
function setupIPC() {
    // Handle file operations
    ipcMain.handle('show-save-dialog', async (event, options) => {
        const result = await dialog.showSaveDialog(mainWindow, options);
        return result;
    });

    ipcMain.handle('show-open-dialog', async (event, options) => {
        const result = await dialog.showOpenDialog(mainWindow, options);
        return result;
    });

    ipcMain.handle('show-message-box', async (event, options) => {
        const result = await dialog.showMessageBox(mainWindow, options);
        return result;
    });

    // Handle external links
    ipcMain.handle('open-external', async (event, url) => {
        await shell.openExternal(url);
    });

    // Handle app info
    ipcMain.handle('get-app-info', () => {
        return {
            name: app.getName(),
            version: app.getVersion(),
            platform: process.platform,
            arch: process.arch,
            electronVersion: process.versions.electron,
            nodeVersion: process.versions.node
        };
    });

    // Handle window controls
    ipcMain.handle('minimize-window', () => {
        mainWindow.minimize();
    });

    ipcMain.handle('maximize-window', () => {
        if (mainWindow.isMaximized()) {
            mainWindow.unmaximize();
        } else {
            mainWindow.maximize();
        }
    });

    ipcMain.handle('close-window', () => {
        mainWindow.close();
    });

    // Handle app updates
    ipcMain.handle('check-for-updates', async () => {
        // Implement update checking logic here
        return { hasUpdate: false, version: app.getVersion() };
    });
}

/**
 * App event handlers
 */
app.whenReady().then(() => {
    // Create splash screen
    createSplashWindow();
    
    // Create main window after a short delay
    setTimeout(() => {
        createMainWindow();
        createMenu();
        setupIPC();
    }, 1500);

    app.on('activate', () => {
        // On macOS, re-create window when dock icon is clicked
        if (BrowserWindow.getAllWindows().length === 0) {
            createMainWindow();
        }
    });
});

app.on('window-all-closed', () => {
    // On macOS, keep app running even when all windows are closed
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('before-quit', (event) => {
    // Handle any cleanup before quitting
    if (mainWindow) {
        mainWindow.webContents.send('app-before-quit');
    }
});

// Security: Prevent new window creation
app.on('web-contents-created', (event, contents) => {
    contents.on('new-window', (event, navigationUrl) => {
        event.preventDefault();
        shell.openExternal(navigationUrl);
    });
});

// Handle certificate errors
app.on('certificate-error', (event, webContents, url, error, certificate, callback) => {
    if (isDev) {
        // In development, ignore certificate errors
        event.preventDefault();
        callback(true);
    } else {
        // In production, use default behavior
        callback(false);
    }
});

// Handle protocol
app.setAsDefaultProtocolClient('autonomousai');

// Handle deep links (Windows/Linux)
app.on('second-instance', (event, commandLine, workingDirectory) => {
    // Someone tried to run a second instance, focus our window instead
    if (mainWindow) {
        if (mainWindow.isMinimized()) mainWindow.restore();
        mainWindow.focus();
    }
});

// Prevent multiple instances
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
    app.quit();
} else {
    app.on('second-instance', () => {
        // Someone tried to run a second instance, focus our window instead
        if (mainWindow) {
            if (mainWindow.isMinimized()) mainWindow.restore();
            mainWindow.focus();
        }
    });
}

// Export for testing
module.exports = { createMainWindow, createMenu };
