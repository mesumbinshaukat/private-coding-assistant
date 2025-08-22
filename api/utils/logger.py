"""
Logging utilities for the AI Agent
Provides structured logging with different levels and formats
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict
from pathlib import Path

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        return json.dumps(log_entry)

class AgentLogger:
    """
    Enhanced logger for the AI Agent with structured logging
    
    Features:
    - JSON structured logging
    - Multiple log levels
    - File and console output
    - Performance tracking
    - Error categorization
    """
    
    def __init__(self, name: str = "autonomous_agent"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup log handlers for console and file output"""
        
        # Console handler with simple format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler with JSON format (for production)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "agent.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        
        # Error file handler
        error_handler = logging.FileHandler(log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional extra data"""
        self.logger.info(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional extra data"""
        self.logger.debug(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional extra data"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional extra data"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional extra data"""
        self.logger.critical(message, extra=kwargs)
    
    def log_request(self, endpoint: str, method: str, user_id: str, duration: float = None):
        """Log API request with timing"""
        extra_data = {
            "endpoint": endpoint,
            "method": method,
            "user_id": user_id,
            "type": "api_request"
        }
        
        if duration is not None:
            extra_data["duration_ms"] = duration * 1000
        
        self.info(f"API Request: {method} {endpoint}", **extra_data)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics"""
        extra_data = {
            "operation": operation,
            "duration_ms": duration * 1000,
            "type": "performance",
            **metrics
        }
        
        self.info(f"Performance: {operation} completed in {duration:.3f}s", **extra_data)
    
    def log_model_operation(self, operation: str, model_name: str, **details):
        """Log model-related operations"""
        extra_data = {
            "operation": operation,
            "model_name": model_name,
            "type": "model_operation",
            **details
        }
        
        self.info(f"Model Operation: {operation} on {model_name}", **extra_data)
    
    def log_training_step(self, step: int, loss: float, metrics: Dict[str, Any]):
        """Log training step with metrics"""
        extra_data = {
            "training_step": step,
            "loss": loss,
            "metrics": metrics,
            "type": "training"
        }
        
        self.info(f"Training Step {step}: loss={loss:.4f}", **extra_data)
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with additional context"""
        extra_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "type": "error_with_context"
        }
        
        self.error(f"Error occurred: {type(error).__name__}: {str(error)}", **extra_data)

# Global logger instance
def setup_logger(name: str = "autonomous_agent") -> logging.Logger:
    """
    Setup and return configured logger
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    agent_logger = AgentLogger(name)
    return agent_logger.logger

# Performance tracking decorator
def log_performance(operation_name: str):
    """Decorator to log function performance"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger = logging.getLogger("autonomous_agent")
                logger.info(
                    f"Performance: {operation_name} completed",
                    extra={
                        "operation": operation_name,
                        "duration_ms": duration * 1000,
                        "type": "performance",
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger = logging.getLogger("autonomous_agent")
                logger.error(
                    f"Performance: {operation_name} failed",
                    extra={
                        "operation": operation_name,
                        "duration_ms": duration * 1000,
                        "type": "performance",
                        "success": False,
                        "error": str(e)
                    }
                )
                
                raise
        
        def sync_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger = logging.getLogger("autonomous_agent")
                logger.info(
                    f"Performance: {operation_name} completed",
                    extra={
                        "operation": operation_name,
                        "duration_ms": duration * 1000,
                        "type": "performance",
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger = logging.getLogger("autonomous_agent")
                logger.error(
                    f"Performance: {operation_name} failed",
                    extra={
                        "operation": operation_name,
                        "duration_ms": duration * 1000,
                        "type": "performance",
                        "success": False,
                        "error": str(e)
                    }
                )
                
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Example usage
if __name__ == "__main__":
    # Test the logger
    logger = setup_logger()
    
    logger.info("Agent starting up")
    logger.debug("Debug information", extra={"component": "test"})
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Test performance logging
    @log_performance("test_operation")
    def test_function():
        import time
        time.sleep(1)
        return "result"
    
    result = test_function()
    print(f"Function result: {result}")
