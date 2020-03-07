import yaml

with open("./config/config.yaml", "r") as f:
    configs = yaml.safe_load(f)

ENABLE_CACHE = configs["CACHE"]
REDIS_CONFIG = configs["redis"]
SERVER_DEBUG = configs["SERVER_DEBUG"]

