from dotmap import DotMap

a = {"a": [1, 2], "b": 2}
b = DotMap(a)

print(b.a)


from config_utils import Config

config_file = "config/test.yml"
config = Config(config_file)
print(config.get("eval_interval"))
print(config.get_dict()["W"])
cfg = config.get_dotmap()
print(cfg.eval_interval)
