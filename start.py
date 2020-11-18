#! /usr/bin/env python3

import requests
import sys

with open("program.py") as f:
    program = f.read()

resp = requests.post(sys.argv[1], headers={"X-API-KEY": sys.argv[2]}, json={"program": program})
print(resp.status_code)
print(resp.text)
