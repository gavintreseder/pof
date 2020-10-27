import configparser
import os
import toml

file_path = os.path.dirname(__file__) + '\\'

# Legacy config parser format
#config = configparser.ConfigParser(defaults=None)
#config.read(file_path + "configs\\config.ini")


config.load(file_path + "configs\\config.toml")