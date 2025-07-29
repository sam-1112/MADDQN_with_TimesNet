import yaml

def loding_yaml():
    """
    Loads a YAML configuration file and returns the configurations as a dictionary.

    :return: Dictionary containing the configurations
    """
    file_path = '/home/bcs113110/BCS113110/DDQN_PSO/TimesNet/config/config.yaml'
    with open(file_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs