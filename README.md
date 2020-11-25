# pof



# PoF
Title (A Title Image too if possible…Edit them on canva.com if you are not a graphic designer.)
## Description

## Demo

## Technical

### Installation

'''
pip install pof
'''
Description(Describe by words and images alike)
Demo(Images, Video links, Live Demo links)
Technologies Used
Special Gotchas of your projects (Problems you faced, unique elements of your project)
Technical Description of your project like- Installation, Setup, How to contribute.

### Get the source code up and running


#### Clone Repo


### Select Interpreter
ctrl + shift + p

https://code.visualstudio.com/docs/python/environments 


#### Setup a virtual environment
Create a new virtual environment so that previous python installations work
>>> virtualenv .venv -p python3.9

Activate the venv or launch a new terminal. You will know this has worked if you have (.venv) in front of your path in the terminal
>>> ./venv/bin/activate

Upgrade pip to make sure everything installs correctly

>>> python -m pip install --upgrade pip

Install the dependencies that are for this module to run

>>> pip install -r requirements.txt

Install the dependencies that are required to develop this module

>>> pip install poetry
>>> poetry install

If using notebooks are not detecting the venv you have set up you may need to link the ipykernel. The symptom to observe is that unittests pass, but the notebooks raise ModuleNotFoundError for dependencies that have been 

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



## Brotli install error

https://visualstudio.microsoft.com/downloads/