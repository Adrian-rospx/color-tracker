#/bin/python

from venv import create
from os.path import join, expanduser, abspath
from subprocess import run

# define environment directory
dir = join(expanduser("~"), ".venv")
# create virtual environment
create(dir, with_pip = True)

# install dependencies from requirements.txt
run(["bin/pip", "install", "-r", abspath("requirements.txt")], cwd = dir)