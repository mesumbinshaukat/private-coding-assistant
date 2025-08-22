# Autonomous AI Agent Desktop App Setup Script
# Installs dependencies and sets up the development environment

param(
    [switch]$Production,
    [switch]$SkipNodeInstall,
    [switch]$Force
)

Write-Host "🤖 Autonomous AI Agent Desktop Setup" -ForegroundColor Cyan
Write-Host "Setting up the desktop application development environment..." -ForegroundColor White

# Function to check if command exists
function Test-Command($command) {
    $null = Get-Command $command -ErrorAction SilentlyContinue
    return $?
}

# Function to get Node.js version
function Get-NodeVersion {
    try {
        $version = node --version 2>$null
        return $version -replace 'v', ''
    }
    catch {
        return $null
    }
}

# Function to compare versions
function Compare-Version($version1, $version2) {
    $v1 = [System.Version]$version1
    $v2 = [System.Version]$version2
    return $v1.CompareTo($v2)
}

# Check prerequisites
Write-Host "`n📋 Checking prerequisites..." -ForegroundColor Yellow

# Check Windows version
$osVersion = [System.Environment]::OSVersion.Version
if ($osVersion.Major -lt 10) {
    Write-Host "❌ Windows 10 or later is required" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Windows version: $($osVersion.Major).$($osVersion.Minor)" -ForegroundColor Green

# Check PowerShell version
$psVersion = $PSVersionTable.PSVersion
if ($psVersion.Major -lt 5) {
    Write-Host "❌ PowerShell 5.0 or later is required" -ForegroundColor Red
    exit 1
}
Write-Host "✅ PowerShell version: $($psVersion.Major).$($psVersion.Minor)" -ForegroundColor Green

# Check Node.js
if (-not $SkipNodeInstall) {
    if (Test-Command node) {
        $nodeVersion = Get-NodeVersion
        if ($nodeVersion -and (Compare-Version $nodeVersion "18.0.0") -ge 0) {
            Write-Host "✅ Node.js version: $nodeVersion" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Node.js version $nodeVersion found, but 18.0.0+ recommended" -ForegroundColor Yellow
            if (-not $Force) {
                $response = Read-Host "Continue anyway? (y/N)"
                if ($response -ne 'y' -and $response -ne 'Y') {
                    Write-Host "Please update Node.js: https://nodejs.org/" -ForegroundColor Yellow
                    exit 1
                }
            }
        }
    } else {
        Write-Host "❌ Node.js not found" -ForegroundColor Red
        Write-Host "Please install Node.js 18+ from: https://nodejs.org/" -ForegroundColor Yellow
        exit 1
    }
}

# Check npm
if (Test-Command npm) {
    $npmVersion = npm --version 2>$null
    Write-Host "✅ npm version: $npmVersion" -ForegroundColor Green
} else {
    Write-Host "❌ npm not found" -ForegroundColor Red
    exit 1
}

# Check Git
if (Test-Command git) {
    $gitVersion = git --version 2>$null
    Write-Host "✅ Git installed: $gitVersion" -ForegroundColor Green
} else {
    Write-Host "⚠️  Git not found - recommended for development" -ForegroundColor Yellow
}

# Create project structure
Write-Host "`n📁 Creating project structure..." -ForegroundColor Yellow

$directories = @(
    "electron-app",
    "electron-app\renderer",
    "electron-app\renderer\css",
    "electron-app\renderer\js",
    "electron-app\renderer\js\components",
    "electron-app\renderer\assets",
    "electron-app\build",
    "vscode-extension",
    "vscode-extension\src",
    "vscode-extension\webviews",
    "vscode-extension\resources",
    "installer",
    "installer\resources"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Gray
    }
}

Write-Host "✅ Project structure created" -ForegroundColor Green

# Install Electron app dependencies
Write-Host "`n📦 Installing Electron app dependencies..." -ForegroundColor Yellow

Set-Location "electron-app"

# Create package.json if it doesn't exist
if (-not (Test-Path "package.json")) {
    Write-Host "Creating package.json..." -ForegroundColor Gray
    
    $packageJson = @{
        name = "autonomous-ai-agent-desktop"
        version = "1.0.0"
        description = "Desktop application for Autonomous AI Agent"
        main = "main.js"
        scripts = @{
            start = "electron ."
            dev = "electron . --dev"
            build = "electron-builder"
            dist = "electron-builder --publish=never"
            test = "jest"
            lint = "eslint ."
        }
        author = "Autonomous AI Agent Team"
        license = "MIT"
        devDependencies = @{
            electron = "^27.0.0"
            "electron-builder" = "^24.6.4"
            "electron-devtools-installer" = "^3.2.0"
            eslint = "^8.50.0"
            jest = "^29.7.0"
        }
        dependencies = @{
            "monaco-editor" = "^0.44.0"
            "material-components-web" = "^14.0.0"
            axios = "^1.5.0"
            "socket.io-client" = "^4.7.2"
        }
        build = @{
            appId = "com.autonomousai.agent.desktop"
            productName = "Autonomous AI Agent"
            directories = @{
                output = "dist"
            }
            files = @(
                "main.js",
                "preload.js",
                "renderer/**/*",
                "package.json"
            )
            win = @{
                target = "nsis"
                icon = "renderer/assets/icon.ico"
            }
            nsis = @{
                oneClick = $false
                allowToChangeInstallationDirectory = $true
                createDesktopShortcut = $true
                createStartMenuShortcut = $true
            }
        }
    } | ConvertTo-Json -Depth 10

    $packageJson | Out-File -FilePath "package.json" -Encoding UTF8
}

# Install dependencies
Write-Host "Installing npm dependencies..." -ForegroundColor Gray
npm install

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install Electron dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Electron dependencies installed" -ForegroundColor Green

# Install VSCode extension dependencies
Write-Host "`n📦 Installing VSCode extension dependencies..." -ForegroundColor Yellow

Set-Location "..\vscode-extension"

# Create VSCode extension package.json
if (-not (Test-Path "package.json")) {
    Write-Host "Creating VSCode extension package.json..." -ForegroundColor Gray
    
    $vscodePackageJson = @{
        name = "autonomous-ai-agent-extension"
        displayName = "Autonomous AI Agent"
        description = "AI-powered coding assistant"
        version = "1.0.0"
        engines = @{
            vscode = "^1.80.0"
        }
        categories = @("Other", "Machine Learning")
        activationEvents = @("*")
        main = "./extension.js"
        contributes = @{
            commands = @(
                @{
                    command = "autonomousai.generateCode"
                    title = "Generate Code"
                    category = "Autonomous AI"
                },
                @{
                    command = "autonomousai.searchWeb"
                    title = "Search Web"
                    category = "Autonomous AI"
                },
                @{
                    command = "autonomousai.reasonProblem"
                    title = "Reason About Problem"
                    category = "Autonomous AI"
                },
                @{
                    command = "autonomousai.openAssistant"
                    title = "Open AI Assistant"
                    category = "Autonomous AI"
                }
            )
            keybindings = @(
                @{
                    command = "autonomousai.generateCode"
                    key = "ctrl+shift+g"
                    when = "editorTextFocus"
                },
                @{
                    command = "autonomousai.openAssistant"
                    key = "ctrl+shift+a"
                }
            )
            views = @{
                explorer = @(
                    @{
                        id = "autonomousai.assistant"
                        name = "AI Assistant"
                        when = "true"
                    }
                )
            }
            configuration = @{
                title = "Autonomous AI Agent"
                properties = @{
                    "autonomousai.apiEndpoint" = @{
                        type = "string"
                        default = "https://your-deployment.vercel.app"
                        description = "API endpoint for the AI agent"
                    }
                    "autonomousai.apiToken" = @{
                        type = "string"
                        default = ""
                        description = "Authentication token for the API"
                    }
                    "autonomousai.autoSuggestions" = @{
                        type = "boolean"
                        default = $true
                        description = "Enable automatic code suggestions"
                    }
                }
            }
        }
        scripts = @{
            "vscode:prepublish" = "npm run compile"
            compile = "tsc -p ./"
            watch = "tsc -watch -p ./"
            test = "npm run compile && node ./out/test/runTest.js"
        }
        devDependencies = @{
            "@types/vscode" = "^1.80.0"
            "@types/node" = "^20.x"
            typescript = "^5.2.0"
            "@vscode/test-electron" = "^2.3.4"
        }
        dependencies = @{
            axios = "^1.5.0"
            "socket.io-client" = "^4.7.2"
        }
    } | ConvertTo-Json -Depth 10

    $vscodePackageJson | Out-File -FilePath "package.json" -Encoding UTF8
}

npm install

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install VSCode extension dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "✅ VSCode extension dependencies installed" -ForegroundColor Green

# Create development configuration
Write-Host "`n⚙️  Creating development configuration..." -ForegroundColor Yellow

Set-Location ".."

# Create .env file for development
if (-not (Test-Path ".env")) {
    @"
# Development Environment Configuration
NODE_ENV=development
API_ENDPOINT=https://your-deployment.vercel.app
API_TOKEN=autonomous-ai-agent-2024
DEBUG=true
LOG_LEVEL=debug
"@ | Out-File -FilePath ".env" -Encoding UTF8

    Write-Host "Created .env configuration file" -ForegroundColor Gray
}

# Create launch configuration for VS Code
$vscodeDir = ".vscode"
if (-not (Test-Path $vscodeDir)) {
    New-Item -ItemType Directory -Path $vscodeDir -Force | Out-Null
}

$launchConfig = @{
    version = "0.2.0"
    configurations = @(
        @{
            name = "Launch Electron App"
            type = "node"
            request = "launch"
            cwd = "`${workspaceFolder}/electron-app"
            program = "`${workspaceFolder}/node_modules/.bin/electron"
            args = @(".")
            env = @{
                NODE_ENV = "development"
            }
        },
        @{
            name = "Launch VSCode Extension"
            type = "extensionHost"
            request = "launch"
            runtimeExecutable = "`${execPath}"
            args = @(
                "--extensionDevelopmentPath=`${workspaceFolder}/vscode-extension"
            )
        }
    )
} | ConvertTo-Json -Depth 10

$launchConfig | Out-File -FilePath "$vscodeDir\launch.json" -Encoding UTF8

Write-Host "✅ Development configuration created" -ForegroundColor Green

# Install global tools
Write-Host "`n🔧 Installing global development tools..." -ForegroundColor Yellow

$globalTools = @(
    "electron",
    "typescript", 
    "@vscode/vsce"
)

foreach ($tool in $globalTools) {
    Write-Host "Installing $tool..." -ForegroundColor Gray
    npm install -g $tool
}

Write-Host "✅ Global tools installed" -ForegroundColor Green

# Setup complete
Write-Host "`n🎉 Setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor White
Write-Host "1. Configure your API endpoint in the .env file" -ForegroundColor Gray
Write-Host "2. Start development: npm run dev (in electron-app directory)" -ForegroundColor Gray
Write-Host "3. Open VSCode and press F5 to debug the extension" -ForegroundColor Gray
Write-Host "4. Build for production: npm run build" -ForegroundColor Gray

Write-Host "`nDevelopment commands:" -ForegroundColor White
Write-Host "  cd electron-app && npm run dev    # Start Electron app in dev mode" -ForegroundColor Gray
Write-Host "  cd electron-app && npm run build  # Build for production" -ForegroundColor Gray
Write-Host "  cd electron-app && npm run dist   # Create installer" -ForegroundColor Gray

Write-Host "`nDocumentation:" -ForegroundColor White
Write-Host "  README.md                         # Full documentation" -ForegroundColor Gray
Write-Host "  https://electronjs.org/docs       # Electron documentation" -ForegroundColor Gray
Write-Host "  https://code.visualstudio.com/api # VSCode extension API" -ForegroundColor Gray

Write-Host "`n✨ Ready to build the future of AI-assisted coding!" -ForegroundColor Cyan
