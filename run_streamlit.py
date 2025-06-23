#!/usr/bin/env python3
"""
Simple launcher script for the Streamlit application.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit application."""
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found in current directory")
        print("💡 Make sure you're running this from the project root")
        return 1
    
    # Check if basic files exist
    required_files = ["fake_looker.py", "chat_gemini.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return 1
    
    print("🚀 Starting Looker Query Generator Streamlit App...")
    print("🌐 The app will open in your browser automatically")
    print("📝 Use Ctrl+C to stop the application")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 