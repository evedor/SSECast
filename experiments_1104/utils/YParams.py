import logging

# --- YAML backend -----------------------------------------------------------
# 原工程用 ruamel.yaml；如果运行环境里没装 ruamel（pip install ruamel.yaml），
# 退回到 PyYAML（yaml.safe_load 同样支持 yaml anchors / merge key `<<:`）。
try:
    from ruamel.yaml import YAML as _YAML

    def _load_yaml(filename):
        with open(filename) as _f:
            return _YAML().load(_f)
except ModuleNotFoundError:
    try:
        import yaml as _pyyaml

        def _load_yaml(filename):
            with open(filename) as _f:
                return _pyyaml.safe_load(_f)
    except ModuleNotFoundError:
        try:
            from ruamel_yaml import YAML as _YAML

            def _load_yaml(filename):
                with open(filename) as _f:
                    return _YAML().load(_f)
        except ModuleNotFoundError as _exc:
            raise ModuleNotFoundError(
                "YParams needs a YAML parser: please install one of "
                "`ruamel.yaml` (recommended: python -m pip install ruamel.yaml) or `PyYAML`."
            ) from _exc
# ----------------------------------------------------------------------------

class YParams():
  """ Yaml file parser """
  def __init__(self, yaml_filename, config_name, print_params=False):
    self._yaml_filename = yaml_filename
    self._config_name = config_name
    self.params = {}

    if print_params:
      print("------------------ Configuration ------------------")

    with open(yaml_filename) as _file:
      _config = _load_yaml(yaml_filename)
      if _config is None or config_name not in _config:
          raise KeyError(f"config '{config_name}' not found in {yaml_filename} (have: {list((_config or {}).keys())})")

      for key, val in _config[config_name].items():
        if print_params: print(key, val)
        if val =='None': val = None

        self.params[key] = val
        self.__setattr__(key, val)

    if print_params:
      print("---------------------------------------------------")

  def __getitem__(self, key):
    return self.params[key]

  def __setitem__(self, key, val):
    self.params[key] = val
    self.__setattr__(key, val)

  def __contains__(self, key):
    return (key in self.params)

  def update_params(self, config):
    for key, val in config.items():
      self.params[key] = val
      self.__setattr__(key, val)

  def log(self):
    logging.info("------------------ Configuration ------------------")
    logging.info("Configuration file: "+str(self._yaml_filename))
    logging.info("Configuration name: "+str(self._config_name))
    for key, val in self.params.items():
        logging.info(str(key) + ' ' + str(val))
    logging.info("---------------------------------------------------")
