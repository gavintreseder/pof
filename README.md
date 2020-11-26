
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

### Setup
To download, view and execute the code base you will need:
- Python: The easiest way to setup python is through anaconda. https://www.anaconda.com/products/individual 
- Visual Studio Code: The IDE for navigating and executing the code. https://code.visualstudio.com/download 
- Git: An application for downloading the current repository from GitHub. https://git-scm.com/downloads 

The setup process to run the application on an Essential Energy computer requires a software request for:
1. Anaconda
2. Visual Studio Code
3. Git

### Installation
To access the development code base the following steps are requried:
1. Clone the repository
2. 
#### Clone Repo

https://github.com/gavintreseder/pof

#### Select Interpreter
ctrl + shift + p

https://code.visualstudio.com/docs/python/environments 


#### Setup a virtual environment
##### Execution only
Python applications 
Create a new virtual environment so that previous python installations work
>>> virtualenv .venv -p python3.9

Activate the venv or launch a new terminal. You will know this has worked if you have (.venv) in front of your path in the terminal
>>> /.venv/bin/activate

Upgrade pip to the latest version to make sure packages install correctly

>>> python -m pip install --upgrade pip

Install the dependencies that are for this module to run

>>> pip install -r requirements.txt

##### Optional steps for development
Install the dependencies that are required to develop this module

>>> pip install poetry
>>> poetry install

In some instances notebooks may not detect the venv and may need to be linked the ipykernel. If the unittests are passing, but executing the notebooks raise ModuleNotFoundError then this is the likely cause

>>> ipython kernel install --user --name=.venv

Note: If this causes issues try looking here
- https://stackoverflow.com/questions/58119823/jupyter-notebooks-in-vscode-does-not-use-active-virtual-environment
- https://jupyter-notebook.readthedocs.io/en/latest/troubleshooting.html#python-environments


### poetry
https://python-poetry.org/docs/

(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -


Adding new packages 
>>> poetry add <package>
>>> poetry add --dev <package>

Creating the requirements.txt
>>> poetry export --output requirements.txt --without-hashes

### Check the unit tests are running correctly

ctrl + shift + p
Configure Test Framework

# Heroku

Login to heroku
>>> heroku login -i

Access the pof-engine
>>> heroku git:remote -a pof-engine
#### Open 

# Gotchas
Special Gotchas of your projects (Problems you faced, unique elements of your project)

Problem: Errors using virtual enrironment
Solution: https://stackoverflow.com/questions/45954528/pip-is-configured-with-locations-that-require-tls-ssl-however-the-ssl-module-in/61429593#61429593
Specifically, I copied the following files from C:\Users\MyUser\Miniconda3\Library\bin to C:\Users\MyUser\Miniconda3\DLLs:

libcrypto-1_1-x64.dll
libcrypto-1_1-x64.pdb
libssl-1_1-x64.dll
libssl-1_1-x64.pdb

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

https://visualstudio.microsoft.com/downloads/
