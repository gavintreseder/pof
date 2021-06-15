# PoF

A project for determining the probability of failure for an asset once a maintenance strategy has been applied.

## Description

TODO

## Demo

Detailed demonstrations can be found in notebooks. Notebook demonstrations include:
Value Lost
System - ~the same as component
Component
Failure Mode
Tasks
Indicators

## Technology
This project relies on open source packages that are freely available online. The key packages used for this project are:
- Dash: Interface in a browser session
- Plotly: Interactive graphs
- Lifelines: Survival analysis
- Reliability: Survival analysis

A full list of the packages can be found in pyproject.toml and requirements.txt

## Installation

Here is a step by step guide on how to install the project. It will get you to the point of having a local running instance and only needs to be completed once per installation.

The following steps only need to be completed once

1. **Software Requirements** - Install the pre-requisite software
2. **IDE (VSCode) Setup** - Setup VSCode (EE's IDE) for python projects
3. **Clone the repository** - Get the code for the pof project from the cloud
4. **Virtual environment Setup** - Get the code

The steps below explain how to complete this process in VS Code. There are two areas where we will be providing inputs to setup; the terminal and the VSCode pallete

Terminal commands are typed into the terminal (typically on the bottom edge of your IDE, but if not can be accessed with ctrl + shift + `) and exectued by pressing enter

`This is a terminal command`

The VSCode Pallete is used to make changes to the IDE and is accessed with ctrl + shift + p. If autocomplete matches the command, you can select this with the mouse.

> This is a command pallete command

## Software Requirements

To download, view and execute the code base you will need:

- **Python**: Python can be downloaded from https://www.python.org/ or alternatively using anacond https://www.anaconda.com/products/individual
- **Git**: An application for connecting to git (version management software) and downloading the current repository from GitHub. https://git-scm.com/downloads
- **Visual Studio Code**: The IDE for navigating and executing the code. https://code.visualstudio.com/download

Note: This software is available via software centre on the Essential Energy network. If it is unavailable here, then a software request will need to be submitted to Etech.

## IDE (VSCode) Setup

The easiest way to navigate and explore this project is through an IDE. VSCode requires several extensions to be enabled to execute python scripts, notebooks and to improve the user experience. Extensions can be installed from the extensions (box icon) in the the activity bar (left hand side). Please install:

- Python: The interpreter using to execute \*.py files and debug the project
- Jupyter: An extension that supports interactive notebooks
- GitLens (optional): Used to manage the repository

VSCode may detect things about your setup during installation (E.g. install data science tools, linting) via blue boxes in the bottom right corner. The exact content of these messages varies, but it is suggested you accept the VSCode recommendation.

After clicking install for Python, Vscode may prompt you to select an interpreter. This should automatically appear in the bottom right hand corner so follow the given instructions. When the Python options appear as a drop down from the Terminal bar, select the correct version of Python that was downloaded (3.9.0 64-bit)

*To set this up manually, click on the blue status bar in the bottom left-hand corner (the interpreter words will be yellow). This will take you up to the terminal, select Python 3.9.0 64-bit.

Essential Energy Laptop option: The default installation at EE does not add python variables to the PATH so you will need to change your default terminal.

ctrl + shift + p
Terminal: Select Default Shell
Command Prompt

**_Validation_**: The dropdown box on the top edge of the terminal ends with "cmd"

## Clone the repository

The code base is hosted on an online repository called github. To obtain the latest version of the source code the repo can be cloned onto your local machine.

ctrl + shift + p (in vs code)

> Git:Clone
> https://github.com/gavintreseder/pof.git

Follow the prompts to save the project to a location on your computer. It is recommended that this is installed in it's own folder as input and output folders will also be created at this location.

**_Validation_**: A file strucuture is visible in the file explorer

## Virtual environment setup

A virtual environment will keep all the requirements for this project in a single location and reduce the chance of any conflicts with other python projects you have.

Create a new virtual environment

`python -m venv .venv`

Once the virtual environment has been created you will need to select it has your interpreter. You will know this has worked when the interpreter in the status bar reflects the venv you have just made (E.g. "Python 3.9.2 64-bit ('.venv':venv)". This can be achieved through one of three methods.

#### Option 1 - Automatic Dectection

VSCode may detect the new virtual environment automatically and prompt you to switch.

#### Option 2 - Status Bar

Select the interpreter from the status bar in the bottom left of the screen

#### Option 3 - Commnad Pallette

Select the interpreter from the command pallette.

ctrl + shift + p

> Python: Select Interpreter
> Python 3.9.2 64-bit ('venv':venv)

The terminal you created the virtual environment from will still be using the local python installation rather than your new venv. Any new terminals you open will activate in this venv by default.

#### Option 4 - Terminal
Run the command

> .venv\Scripts\activate

**_Validation_**: Your terminal path starts with (.venv) C:\ ....

### Dependencies

This python project relies on dependencies (open source pacakges) that need to be installed. Two simple methods for this are using pip or poetry.

#### Option 1 - Poetry

Install the dependencies that are required to develop this module

`pip install poetry`

then

`poetry install`

Note: If the installation does not finish it can be caused by OneDrive locking files during upload. If this occurs attempt the install again.

If an error occurs about ‘Build Code’ not installed, it may be due to C++ not installed on the computer.  If so, follow the link provided in the error message or use the following link https://visualstudio.microsoft.com/vs/ to find and install ‘Desktop Development with C++’.  You will need to restart the computer for this to happen. After installing this program, repeat ‘To develop the project (optional)’ step.

In some instances, notebooks may not detect the venv and may need to be linked the ipykernel. If the unittests are passing, but executing the notebooks raise ModuleNotFoundError then this is the likely cause

>>> ipython kernel install --user --name=.venv

Note: If this causes issues see gotchas

#### Option 2 - Pip

Pip is distributed with python by default; however, it receives regular updates so it advisable to install latest version before using.

`python -m pip install --upgrade pip`

Now install the dependencies that are for this module to run. During the installation of these packages, red or yellow warnings indicate a package has not installed correctly. (Note: VSCode cannot install the pacakges when connected to the Essential Energy network or on the F5 VPN)

`pip install -r requirements.txt`

### Configure the test framework

The test framework can be configured from the command pallette
ctrl + shift + p

> Python: Configure Tests
> unittest
> test
> test\_\*.py

Once configured, disover tests (refresh icon) to populate a list of tests for this project.

To execute the tests:
- select the test icon in the activity bar (it resembles a beaker)
- Run all tests (the green double arrow in the top left corner)

## Usage

To run the interface, open the file main.py in the pof folder, and hit the play button at the top right hand corner of VS Code.

After executing main.py, the application can be opened at http://127.0.0.1:8050/. Google chrome is the preferred browser.

## Notebooks

Notebooks have been developed as documentation to explain how various parts of the project have been developed. To execute a notebook select a notebook (\*.ipynb) from the explorer, and confirm you trust the notebook

Hint: It is recommended you trust all notebooks. When prompted select "Trust all notebooks" and tick the trust all notebooks box in the settings window that opens. You can close the window

**_Validation_**: The top right corner of your notebook window will indicate that it is Trusted

# ADDITIONAL #

## Poetry

Poetry is a package manager for python projects https://python-poetry.org/docs/ that has been used to manage package dependencies in this project

Adding new packages

`poetry add <package>`

Adding new packages which have development dependencies

`poetry add --dev <package>`

Updating packages

`poetry update`

Creating the requirements.txt

`poetry export --output requirements.txt --without-hashes`

## Heroku

Login to heroku

`heroku login -i`

Access the pof-engine

`heroku git:remote -a pof-engine`

## Gotchas

Here is a collection of unique problems I faced at different times during the project and how I solved the issues

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

https://opensource.com/article/18/2/why-python-devs-should-use-pipenv

## Ipykernel issue

ipykernel issue:
ipython kernel install --user --name=projectname
https://anbasile.github.io/posts/2017-06-25-jupyter-venv/

## Brotli install error

Ensure you have the visual studio C++ base tools installed
https://visualstudio.microsoft.com/downloads/

## Jupter Notebooks

In some instances notebooks may not detect the venv and may need to be linked the ipykernel. If the unittests are passing, but executing the notebooks raises ModuleNotFoundError then this is the likely cause

`ipython kernel install --user --name=.venv`
