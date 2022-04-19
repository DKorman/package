import yaml
import argparse

def parse_config():
    """
    Method that parses the Project config file

    Returns: config dictionary
    """

    def _parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument('configuration_path', type=str)

        return parser.parse_args()

    def _load_config_file():
        return yaml.safe_load(open(_parse_arguments().configuration_path, encoding='utf-8'))

    config = _load_config_file()

    return config
