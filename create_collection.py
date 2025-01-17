from generator import Generator
from easydict import EasyDict
import ipdb
import yaml

def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    return EasyDict(config_data)

def get_generator():
	config_path = 'config.yaml'
	config = read_yaml_config(config_path)
	generator = Generator(config)
	return generator

if __name__ == '__main__':
    ipdb.set_trace()
    generator = get_generator()
    generator.create_vector_db()
