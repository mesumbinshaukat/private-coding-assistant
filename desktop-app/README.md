# Autonomous AI Agent Desktop Application

A modern Electron-based desktop application that integrates with the Autonomous AI Agent API for code generation, web search, and reasoning tasks.

## Features

- **Code Generation**: Generate code in multiple programming languages with intelligent templates
- **Web Search**: Perform deep web searches using DuckDuckGo API
- **Step-by-Step Reasoning**: Get detailed analysis and solutions for complex problems
- **Real-time API Status**: Monitor API health and availability
- **Code Saving**: Save generated code to local files
- **Modern UI**: Clean, responsive interface with tabbed navigation

## Screenshots

The application features a clean, modern interface with:
- Gradient header with real-time status indicator
- Tabbed navigation for different features
- Input forms with validation
- Rich output display with syntax highlighting
- File save functionality

## Prerequisites

- **Node.js** (v16 or higher)
- **npm** (comes with Node.js)
- **Windows 10/11** (primary target platform)

## Installation

### Quick Setup (Windows)

1. **Clone or download** the project files
2. **Run the setup script**:
   ```bash
   setup.bat
   ```

### Manual Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the application**:
   ```bash
   npm start
   ```

## Usage

### Starting the Application

```bash
npm start
```

### Building the Application

```bash
# Create a distributable package
npm run build

# Create a directory with the built app
npm run pack
```

### Using the Features

#### Code Generation
1. Navigate to the "Code Generation" tab
2. Enter your code request (e.g., "Write a Python function for fibonacci sequence")
3. Select the programming language
4. Click "Generate Code"
5. Review the generated code and save it if needed

#### Web Search
1. Go to the "Web Search" tab
2. Enter your search query
3. Select search depth (3, 5, or 10 results)
4. Click "Search"
5. Browse through the results and synthesized answers

#### Reasoning
1. Visit the "Reasoning" tab
2. Describe the problem you want analyzed
3. Choose the domain (coding, algorithms, etc.)
4. Toggle mathematical analysis if needed
5. Click "Analyze"
6. Review the step-by-step reasoning and solution

#### API Status
1. Check the "API Status" tab
2. View real-time API health information
3. Monitor available features and deployment status

## API Integration

The desktop application connects to the Autonomous AI Agent API at:
```
https://private-coding-assistant.vercel.app
```

### Authentication
- Uses JWT token authentication
- Token is configured in the main process
- All API calls include proper authorization headers

### Endpoints Used
- `GET /health` - Health check
- `POST /generate` - Code generation
- `POST /search` - Web search
- `POST /reason` - Reasoning analysis
- `GET /status` - API status

## Development

### Project Structure
```
desktop-app/
├── main.js          # Main Electron process
├── preload.js       # Preload script for secure IPC
├── index.html       # Main application interface
├── styles.css       # Application styles
├── renderer.js      # Renderer process logic
├── package.json     # Dependencies and scripts
├── setup.bat        # Windows setup script
└── README.md        # This file
```

### Key Technologies
- **Electron** - Cross-platform desktop app framework
- **HTML/CSS/JavaScript** - Frontend interface
- **Node.js** - Backend runtime
- **Axios** - HTTP client for API calls

### Security Features
- Context isolation enabled
- Node integration disabled
- Secure IPC communication
- Input validation and sanitization

## Troubleshooting

### Common Issues

#### "Module not found" errors
- Ensure all dependencies are installed: `npm install`
- Check Node.js version compatibility

#### API connection failures
- Verify the API is running at the configured URL
- Check network connectivity
- Ensure authentication token is valid

#### Build failures
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall: `rm -rf node_modules && npm install`

### Debug Mode
To enable debug mode with DevTools:
```bash
set NODE_ENV=development
npm start
```

## Configuration

### API Settings
Edit `main.js` to modify:
- API base URL
- Authentication token
- Request timeouts

### UI Customization
Modify `styles.css` to change:
- Color scheme
- Layout dimensions
- Typography

## Building for Distribution

### Windows
```bash
npm run build
```
Creates a Windows installer (.exe) in the `dist` folder.

### Cross-platform
The application can be built for:
- Windows (NSIS installer)
- macOS (DMG)
- Linux (AppImage)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Open an issue in the repository

## Changelog

### v1.0.0
- Initial release
- Code generation with multiple languages
- Web search integration
- Step-by-step reasoning
- Real-time API status monitoring
- File save functionality
- Modern, responsive UI
