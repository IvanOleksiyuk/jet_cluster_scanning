# this code was maily copied from Jonny Jacksons blog
# https://jonnyjxn.medium.com/how-to-config-your-machine-learning-experiments-without-the-headaches-bb379de1b957
# With some modifications added like a DotMap
import yaml
from dotmap import DotMap


def merge_dictionaries_recursively(dict1, dict2):
    """Update two config dictionaries recursively.
    Args:
      dict1 (dict): first dictionary to be updated
      dict2 (dict): second dictionary which entries should be preferred
    """
    if dict2 is None:
        return

    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            merge_dictionaries_recursively(dict1[k], v)
        else:
            dict1[k] = v


class Config(object):
    """Simple dict wrapper that adds a thin API allowing for slash-based
    retrieval of nested elements, e.g. cfg.get_config("meta/dataset_name")
    """

    def __init__(self, config_path, default_path=None):
        with open(config_path) as cf_file:
            cfg = yaml.safe_load(cf_file.read())

        if default_path is not None:
            with open(default_path) as def_cf_file:
                default_cfg = yaml.safe_load(def_cf_file.read())
            merge_dictionaries_recursively(default_cfg, cfg)

        self._data = cfg

    def get(self, path=None, default=None):
        # we need to deep-copy self._data to avoid over-writing its data
        sub_dictionary = dict(self._data)

        if path is None:
            return sub_dictionary

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dictionary = sub_dictionary.get(path_item)

            value = sub_dictionary.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default

    def get_dotmap(self):
        return DotMap(self._data)
