import yaml
import os
def loding_yaml():
    """
    Loads a YAML configuration file and returns the configurations as a dictionary.

    :return: Dictionary containing the configurations
    """
    file_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(file_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs