"""Virtual environment creation script. Run with:

```bash
python setup.py
```
"""

import os
import sys
import subprocess
import venv

def create_venv(venv_dir: str = ".venv", requirements_file: str = "requirements.txt"):
    
    # Create venv
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment in {venv_dir}...")
        venv.create(venv_dir, with_pip = True)
    else:
        print(f"Virtual environment already exists at {venv_dir}")

    # Find correct pip executable
    if os.name == "nt": 
        # for windows
        pip_executable = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        # for linux / macos
        pip_executable = os.path.join(venv_dir, "bin", "pip")

    # Install requirements
    if os.path.exists(requirements_file):
        print(f"Installing dependencies from {requirements_file}...")
        subprocess.check_call([pip_executable, "install", "-r", requirements_file])
    else:
        print(f"No {requirements_file} found. Skipping installation...")

    print("Setup complete")

if __name__ == "__main__":
    create_venv()
