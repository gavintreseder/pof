import configparser

config = configparser.ConfigParser(defaults=None)
config.read("configs\\config.ini")
config.read("configs\\test_config.ini")