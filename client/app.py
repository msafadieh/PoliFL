#! /usr/bin/env python3
import os
from io import BytesIO
from threading import Thread
from uuid import uuid4

import certifi
import dill
from flask import Flask, make_response, request
import pycurl
import requests

import ancile
from ancile.core.core import execute

NODEKEY = os.environ["NODE_KEY"]
WEBROOT = os.environ.get("WEBROOT", "/var/www/html/models")
WEBPATH = os.environ.get("WEBPATH", "/models/")

app = Flask(__name__)

def execute_program(model_url, callback_url, program):
    print(f"Fetching model {model_url}")
#    byte_buffer = BytesIO()
#    curl = pycurl.Curl()
#    curl.setopt(curl.URL, model_url)
#    curl.setopt(curl.WRITEDATA, byte_buffer)
#    curl.setopt(curl.CAINFO, certifi.where())
#    curl.perform()
    resp = requests.get(model_url).content
    dpp = dill.loads(resp)
#    curl.close()
    
    res = execute(users_secrets=[],
                  program=program,
                  app_id=None,
                  app_module=None,
                  data_policy_pairs=[dpp])
    if res["result"] == "error":
        message = {"status": "ERROR", "error": res["traceback"]}
    elif not res["data"]:
        message = {"status": "ERROR", "error": "No DPP returned"}
    else:
        uuid = str(uuid4())
        with open(os.path.join(WEBROOT, uuid), "wb+") as f:
            dill.dump(res["data"], f)
            message = {"status": "OK", "data_policy_pair": "{}{}".format(WEBPATH, uuid)}
    print(message)
    res = requests.post(callback_url, json=message, headers={"X-API-Key": NODEKEY})
    print(res.text)

@app.route("/status")
def status():
    return make_response("OK", 200)

@app.route("/execute", methods=["POST"])
def execute_view():
    json = request.json

    if json:
        print(json)
        Thread(target=execute_program, kwargs=json).start()
        return make_response("OK", 200)
    return make_response("BAD", 500)

