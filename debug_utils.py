# debug_utils.py
import os
import subprocess
import requests
import psutil
import time
import sys


def check_internet_connection():
    """Check if internet connection is available."""
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.RequestException:
        return False


def check_library_installation(lib_name):
    """Check if a library is installed."""
    try:
        __import__(lib_name)
        return True
    except ImportError:
        return False


def attempt_library_install(lib_name):
    """Attempt to install or update a library."""
    if check_internet_connection():
        try:
            subprocess.run([sys.executable,
                            '-m',
                            'pip',
                            'install',
                            '--upgrade',
                            lib_name],
                           check=True,
                           capture_output=True,
                           text=True)
            return True
        except subprocess.CalledProcessError as e:
            return False
    return False


def check_system_resources():
    """Check system memory and CPU usage."""
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent()
    if memory.percent > 90 or cpu > 90:
        return True
    return False


def self_diagnosis():
    """Perform self-diagnosis of the environment."""
    issues = []
    if not os.path.exists('requirements.txt'):
        issues.append("No 'requirements.txt' file found.")
    else:
        with open('requirements.txt', 'r') as f:
            content = f.read().lower()
            if 'google-generative-ai' not in content:
                issues.append(
                    "'google-generative-ai' missing in requirements.txt.")
            if 'openai' not in content:
                issues.append("'openai' missing in requirements.txt.")
            if 'anthropic' not in content:
                issues.append("'anthropic' missing in requirements.txt.")
    for file in ['bot.py', 'advanced_bot.py', 'main.py']:
        if file in os.listdir():
            try:
                with open(file, 'r') as f:
                    if 'MultiAIClient' not in f.read():
                        issues.append(
                            f"{file} may lack MultiAIClient integration.")
            except Exception as e:
                issues.append(f"Error reading {file}: {str(e)}.")
    if not check_internet_connection():
        issues.append("No internet connection detected.")
    return issues
