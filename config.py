import configparser
import os
import toml

file_path = os.path.dirname(__file__) + r"/"

# Legacy config parser format
# config = configparser.ConfigParser(defaults=None)
# config.read(file_path + "configs\\config.ini")


config = toml.load(file_path + r"configs/config.toml")

if __name__ == "__main__":
    print(config)
    print("Config - Ok")