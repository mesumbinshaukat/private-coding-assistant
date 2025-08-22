# Autonomous AI Agent Desktop Application

A Windows desktop application that integrates with the Autonomous AI Agent API, providing a VSCode-like interface with built-in AI assistance for coding tasks.

## Features

- **Integrated Code Editor**: Modified VSCode with AI integration
- **AI Assistant Sidebar**: Direct interaction with the autonomous agent
- **Real-time Code Analysis**: Automatic complexity and optimization suggestions
- **Web Search Integration**: In-editor search for coding solutions
- **Training Dashboard**: Monitor and trigger agent self-training
- **Multi-language Support**: Python, JavaScript, and more

## Architecture

```
desktop-app/
├── electron-app/           # Electron wrapper application
│   ├── main.js            # Main Electron process
│   ├── renderer/          # Renderer process files
│   ├── preload.js         # Preload script for security
│   └── package.json       # Electron app configuration
├── vscode-extension/       # VSCode extension for AI integration
│   ├── extension.js       # Main extension file
│   ├── package.json       # Extension manifest
│   ├── src/               # Extension source code
│   └── webviews/          # Custom webview panels
├── installer/             # Windows installer files
│   ├── installer.nsi      # NSIS installer script
│   └── resources/         # Installer resources
├── setup.ps1              # Windows setup script
└── build.js               # Build automation script
```

## Installation

### Option 1: Download Pre-built Installer
1. Download `AutonomousAI-Setup.exe` from the releases page
2. Run the installer as administrator
3. Follow the installation wizard
4. Launch from Start Menu or Desktop shortcut

### Option 2: Build from Source

#### Prerequisites
- Node.js 18+ and npm
- Python 3.8+
- Git
- Visual Studio Build Tools (for native modules)

#### Build Steps

1. **Clone the repository**:
   ```powershell
   git clone <repository-url>
   cd private-model/desktop-app
   ```

2. **Run the setup script**:
   ```powershell
   .\setup.ps1
   ```

3. **Build the application**:
   ```powershell
   npm run build
   ```

4. **Create installer**:
   ```powershell
   npm run dist
   ```

## Configuration

### API Endpoint Setup

1. Open the application
2. Go to **Settings** → **AI Agent Configuration**
3. Enter your API endpoint: `https://your-deployment.vercel.app`
4. Enter your authentication token: `autonomous-ai-agent-2024`
5. Test connection and save

### Default Configuration

```json
{
  "api": {
    "endpoint": "https://your-deployment.vercel.app",
    "token": "autonomous-ai-agent-2024",
    "timeout": 30000
  },
  "editor": {
    "theme": "dark",
    "fontSize": 14,
    "tabSize": 4,
    "autoSave": true
  },
  "ai": {
    "autoSuggestions": true,
    "complexityAnalysis": true,
    "realTimeSearch": false
  }
}
```

## Usage

### AI Assistant Panel

The AI Assistant panel provides direct access to all agent capabilities:

1. **Code Generation**:
   - Type your requirements in natural language
   - Select programming language
   - Get generated code with explanations

2. **Code Analysis**:
   - Right-click on code → "Analyze with AI"
   - Get complexity analysis and optimization suggestions
   - View mathematical explanations of algorithms

3. **Web Search**:
   - Search for coding solutions directly in the editor
   - Get synthesized answers from multiple sources
   - Insert code examples with attribution

4. **Step-by-Step Reasoning**:
   - Input complex problems
   - Get detailed solution breakdowns
   - Understand algorithmic approaches

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+A` | Open AI Assistant |
| `Ctrl+Shift+G` | Generate Code |
| `Ctrl+Shift+S` | Search Web |
| `Ctrl+Shift+R` | Reason About Problem |
| `Ctrl+Shift+T` | Trigger Training |
| `F1` | Command Palette |

### Context Menu Integration

Right-click on code to access:
- **Explain Code**: Get AI explanation of selected code
- **Optimize Code**: Get optimization suggestions
- **Generate Tests**: Create test cases automatically
- **Debug Code**: Get debugging assistance
- **Search Similar**: Find similar code patterns online

## Development

### Project Structure

```
electron-app/
├── main.js                # Main Electron process
├── package.json           # Dependencies and scripts
├── renderer/
│   ├── index.html         # Main application window
│   ├── css/
│   │   ├── main.css       # Main stylesheet
│   │   └── ai-panel.css   # AI panel styling
│   ├── js/
│   │   ├── main.js        # Main renderer logic
│   │   ├── ai-client.js   # AI API client
│   │   ├── editor.js      # Editor integration
│   │   └── components/    # UI components
│   └── assets/            # Images and icons
├── preload.js             # Preload script
└── build/                 # Build output
```

### VSCode Extension Structure

```
vscode-extension/
├── package.json           # Extension manifest
├── extension.js           # Main extension file
├── src/
│   ├── ai-provider.js     # AI service provider
│   ├── commands.js        # Command implementations
│   ├── panels.js          # Custom panels
│   └── utils.js           # Utility functions
├── webviews/
│   ├── ai-assistant.html  # AI assistant panel
│   ├── training-dash.html # Training dashboard
│   └── search-panel.html  # Search interface
└── resources/             # Extension resources
```

### Development Commands

```powershell
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Create installer
npm run dist

# Debug Electron app
npm run debug
```

### API Integration

The desktop app communicates with the Vercel-deployed API:

```javascript
// AI API Client
class AIClient {
    constructor(endpoint, token) {
        this.endpoint = endpoint;
        this.token = token;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }

    async generateCode(prompt, language = 'python', context = null) {
        const response = await fetch(`${this.endpoint}/generate`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ prompt, language, context })
        });
        return await response.json();
    }

    async searchWeb(query, depth = 5, includeCode = true) {
        const response = await fetch(`${this.endpoint}/search`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ 
                query, 
                depth, 
                include_code: includeCode 
            })
        });
        return await response.json();
    }

    async reasonStepByStep(problem, domain = 'coding', includeMath = true) {
        const response = await fetch(`${this.endpoint}/reason`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ 
                problem, 
                domain, 
                include_math: includeMath 
            })
        });
        return await response.json();
    }

    async triggerTraining(datasetName = null, trainingType = 'rlhf', iterations = 10) {
        const response = await fetch(`${this.endpoint}/train`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ 
                dataset_name: datasetName,
                training_type: trainingType,
                iterations 
            })
        });
        return await response.json();
    }

    async getStatus() {
        const response = await fetch(`${this.endpoint}/status`, {
            method: 'GET',
            headers: this.headers
        });
        return await response.json();
    }
}
```

## Security Considerations

### Electron Security
- Content Security Policy (CSP) enabled
- Node integration disabled in renderer
- Context isolation enabled
- Preload scripts for secure API access

### API Security
- JWT token authentication
- HTTPS-only communication
- Input validation and sanitization
- Rate limiting protection

### Code Execution
- No local code execution (all processing on server)
- Sandboxed editor environment
- Safe file handling

## Troubleshooting

### Common Issues

1. **Application won't start**:
   - Check Windows version compatibility (Windows 10+)
   - Run as administrator
   - Check antivirus software blocking

2. **Cannot connect to API**:
   - Verify internet connection
   - Check API endpoint URL
   - Validate authentication token
   - Check firewall settings

3. **Code editor not loading**:
   - Clear application cache: `%APPDATA%\AutonomousAI\`
   - Restart application
   - Check for corrupted installation

4. **Performance issues**:
   - Close unnecessary applications
   - Check available RAM (minimum 4GB recommended)
   - Update graphics drivers

### Debug Information

Enable debug mode by:
1. Open Developer Tools: `Ctrl+Shift+I`
2. Go to Console tab
3. Look for error messages
4. Report issues with console output

### Log Files

Application logs are stored in:
- Windows: `%APPDATA%\AutonomousAI\logs\`
- Files: `main.log`, `renderer.log`, `api.log`

## Performance Optimization

### Recommended System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 | Windows 11 |
| RAM | 4 GB | 8 GB+ |
| Storage | 2 GB free | 5 GB+ free |
| Network | Broadband | High-speed |
| GPU | DirectX 11 | DirectX 12 |

### Performance Tips

1. **Close unused tabs**: Each editor tab uses memory
2. **Limit search depth**: Reduce search depth for faster results
3. **Disable real-time features**: Turn off real-time analysis for large files
4. **Use local cache**: Enable caching for frequently used content
5. **Update regularly**: Keep the application updated for performance improvements

## Updates and Maintenance

### Automatic Updates
- Application checks for updates on startup
- Downloads and installs updates automatically
- Backup configuration before updates

### Manual Updates
1. Download latest installer
2. Close running application
3. Run installer (will update existing installation)
4. Restart application

### Configuration Backup
- Settings automatically backed up before updates
- Manual backup: Copy `%APPDATA%\AutonomousAI\config\`
- Restore: Paste configuration files to same location

## Support and Feedback

### Getting Help
- **Documentation**: Check this README and online docs
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: support@autonomous-ai-agent.com

### Feature Requests
1. Open GitHub Issue with label "enhancement"
2. Describe the desired feature
3. Explain use case and benefits
4. Community voting determines priority

### Contributing
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request
5. Participate in code review

## License

This desktop application is part of the Autonomous AI Agent project and is licensed under the MIT License. See the main project LICENSE file for details.

## Acknowledgments

- **Electron**: Cross-platform desktop apps with web technologies
- **Monaco Editor**: The code editor that powers VS Code
- **Material Design**: UI design principles and components
- **VS Code**: Inspiration for editor interface and features
