#! /usr/bin/env python3
import os
from io import BytesIO
from threading import Thread

import certifi
import dill
from flask import Flask, make_response, request
import pycurl
import requests

import ancile
from ancile.core.core import execute

WEBROOT = os.environ.get("WEBROOT", "/var/www/html")
WEBPATH = os.environ.get("WEBPATH", "/")

app = Flask(__name__)

def execute_program(model_url, callback_url, program):
    byte_buffer = BytesIO()
    curl = pycurl.Curl()
    curl.setopt(curl.URL, model_url)
    curl.setopt(curl.WRITEDATA, str_buffer)
    curl.setopt(curl.CAINFO, certifi.where())
    curl.perform()
    curl.close()
    pickled_dpp = byte_buffer.getvalue()
    if pickled_dpp:
        dpp = dill.loads(pickled_dpp)
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
        with open(os.path.join(WEBROOT, uuid), "wb") as f:
            dill.dump(f, res["data"])
            message = {"status": "OK", "data_policy_pair": "{}{}".format(WEBPATH, uuid)}
    print(message)
    res = requests.post(callback_url, json=message)
    print(res.text)

@app.route("/status")
def status():
    return make_response("OK", 200)

@app.route("/execute")
def execute():
    json = request.json

    if json:
        Thread(target=execute_program, kwargs=json).start()
        return make_response("OK", 200)
    return make_response("BAD", 500)

