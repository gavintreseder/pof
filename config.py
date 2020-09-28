import configparser
import os

file_path = os.path.dirname(__file__) + '\\'

config = configparser.ConfigParser(defaults=None)
config.read(file_path + "configs\\config.ini")
