#! /usr/bin/env python3

import os
import socket
from urllib.parse import urlparse
import requests

TEMPLATE = os.environ.get("TEMPLATE", "/data/wg.tmpl")

NODE_NAME = os.environ["NODE_NAME"]
NODE_KEY = os.environ["NODE_KEY"]
SERVER_ENDPOINT = os.environ["SERVER_ENDPOINT"]
PRIVATE_KEY = os.environ["PRIVATE_KEY"]
PUBLIC_KEY = os.environ["PUBLIC_KEY"]
CONFIG_PATH = os.environ["CONFIG_PATH"]

def main():
    resp = requests.put(SERVER_ENDPOINT + "/nodes/" + NODE_NAME,
                        json={"public_key": PUBLIC_KEY},
                        headers={"X-API-Key": NODE_KEY}).json()

    ip_addr = socket.gethostbyname(urlparse(SERVER_ENDPOINT).hostname)
    resp["endpoint"] = "{}:{}".format(ip_addr, resp["port"])
    resp["private_key"] = PRIVATE_KEY

    with open(TEMPLATE) as f:
        template = f.read()

    with open(CONFIG_PATH, 'w') as f:
        f.write(template.format(**resp))

if __name__ == "__main__":
    main()

