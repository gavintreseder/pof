"""
Prepare the code for testing
"""

import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

# https://stackoverflow.com/questions/58062521/no-tests-discovered-when-using-vscode-python-and-absolute-imports
