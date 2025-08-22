"""
Minimal API entry point for Vercel deployment - Debug Version
"""

from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        if self.path == '/':
            response = {"Hello": "World"}
        elif self.path == '/health':
            response = {"status": "ok"}
        else:
            response = {"error": "Not found"}
        
        self.wfile.write(json.dumps(response).encode())
        return