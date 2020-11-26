
# PoF
Title (A Title Image too if possible…Edit them on canva.com if you are not a graphic designer.)
## Description
Description(Describe by words and images alike)
## Demo
Demo(Images, Video links, Live Demo links)


Larger more detailed demonstrations can be found in notebooks/

## Technology
The key packages used for this project are:
- Dash: Interface in a browser session
- Plotly: Interactive graphs
- Lifelines: Survival analysis
- Reliability: Survival analysis

A full like of the packages can be found in pyproject.toml. 
## Technical



# Installation
Here is a step by step guide on how to install the pof-engine. It will get you to the point of having a local running instance.
To access the development code base the following steps are requried:
1. Install requirements
2. Setup VSCode
3. Clone the repository
4. Setup a virtual environment

The steps below explain how to complete this process in VS Code
## Requirements
To download, view and execute the code base you will need:
- Python: The easiest way to setup python is through anaconda. https://www.anaconda.com/products/individual 
- Git: An application for downloading the current repository from GitHub. https://git-scm.com/downloads 
- Visual Studio Code: The IDE for navigating and executing the code. https://code.visualstudio.com/download 

Note: The setup process to run the application on an Essential Energy computer requires a software request for:
1. Anaconda
2. Git
3. Visual Studio Code

Once this has been approved they will be available for installation through the software centre

### VSCode Setup
VSCode requires several extensions to be enabled. The following extensions can be installed by searching and installing from the activity bar (The box icon on the left hand side):
- Python

The default installation at EE does not add python variables to the PATH so will need to change your default terminal

ctrl + shift + p
Terminal: Select Default Shell
Command Prompt

Launch a new terminal through the ribbon or via the shortcut (ctrl + shift + `)
## Clone the repository
The code base is hosted on an online repository called github. To obtain the latest version of the source code the repo can be cloned onto your local machine.

In VScode:
ctrl + shift + p
Git:Clone
https://github.com/gavintreseder/pof


## Setup a virtual environment
A virtual environment will keep all the requirements for this project in a single location and reduce the chance of any conflicts with other python projects you have. 

Create a new virtual environment:
>>> python -m venv .venv
#>>> virtualenv .venv -p python3.9

Activate to the new virtual environment and launch a new terminal for this environment:

ctrl + shift + p
Python: Select Interpreter
Python .... venv)

You will know this has worked if you have (.venv) in front of your path in the terminal and if the status bar shows ('.venv':venv)

Upgrade pip to the latest version to make sure packages install correctly

>>> python -m pip install --upgrade pip

### To execute the project
Install the dependencies that are for this module to run

>>> pip install -r requirements.txt

### To develop the project (optional)
Install the dependencies that are required to develop this module

>>> pip install poetry
>>> poetry install

In some instances notebooks may not detect the venv and may need to be linked the ipykernel. If the unittests are passing, but executing the notebooks raise ModuleNotFoundError then this is the likely cause

>>> ipython kernel install --user --name=.venv

Note: If this causes issues see gotchas
## Unittests
This package is distributed with unittests to validate key functionality. The unittests will often be discovered automatically by VSCode. If this doesn't happen you will need to configure the test framework

ctrl + shift + p
Python: Configure Tests
unittest
test
test_*.py

To execute the tests:
- select the test icon in the activity bar (it resembles a beaker)
- Run all tests

# Usage

## Dash

# Heroku

Login to heroku
>>> heroku login -i

Access the pof-engine
>>> heroku git:remote -a pof-engine
#### Open 

# Dev notes
#### Poetry 
Poetry is a package manager for python projects https://python-poetry.org/docs/

It can be installed on the entire system using:

(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -

Adding new packages 
>>> poetry add <package>
>>> poetry add --dev <package>

Creating the requirements.txt
>>> poetry export --output requirements.txt --without-hashes
# Gotchas
Special Gotchas of your projects (Problems you faced, unique elements of your project)

## Ipykernel
Sometimes the ipykernel will point to your base environment rather than the virtual environment. See these solutions:
- https://stackoverflow.com/questions/58119823/jupyter-notebooks-in-vscode-does-not-use-active-virtual-environment
- https://jupyter-notebook.readthedocs.io/en/latest/troubleshooting.html#python-environments

## SSL Module error
Problem: Errors using virtual enrironment
Solution: https://stackoverflow.com/questions/45954528/pip-is-configured-with-locations-that-require-tls-ssl-however-the-ssl-module-in/61429593#61429593
Specifically, I copied the following files from C:\Users\MyUser\Miniconda3\Library\bin to C:\Users\MyUser\Miniconda3\DLLs:

libcrypto-1_1-x64.dll
libcrypto-1_1-x64.pdb
libssl-1_1-x64.dll
libssl-1_1-x64.pdb

## Pywin32 errors
https://stackoverflow.com/questions/58612306/how-to-fix-importerror-dll-load-failed-while-importing-win32api
Reinstall Pywin32
copy dlls across from other working pacakge. error apparent when running pywin32_postinstall.py -install


https://pipenv-fork.readthedocs.io/en/latest/ 

How to get the virtual environment working - https://medium.com/@vladbezden/new-python-project-configuration-with-vs-code-b41b77f7aed8

Structure - https://docs.python-guide.org/writing/structure/ 

## Brotli install error
Ensure you have the visual studio C++ base tools installed
https://visualstudio.microsoft.com/downloads/