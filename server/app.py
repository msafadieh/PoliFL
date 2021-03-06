from functools import wraps
from queue import Empty, Queue
from threading import Thread
from uuid import uuid4

from flask import Flask, make_response, request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ancile.core.primitives import DataPolicyPair
from server.ancile import execute_program
from server.config import config
from server.models import Application, Base, Node, Policy
from server.networking import start_wireguard, add_peer, peer_down

engine = create_engine(config['SQLALCHEMY_DATABASE_URI'])
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

jobs = {}

app = Flask(__name__)

if config["WG_ON"]:
    with open(config["PRIVKEY_PATH"]) as f:
        start_wireguard(f.read().strip(), [(u.public_key, u.allowed_ips, ) for u in session.query(Node).all() if u.public_key])


@app.route("/")
def index():
    return make_response("Welcome to ancile!")

def make_403():
    return make_response("Unauthorized", 403)

def make_404():
    return make_response("Not found", 404)

def is_admin(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if request.method == "PUT" or request.headers.get("X-API-Key") == config["API_KEY"]:
            return func(*args, **kwargs)
        return make_403()
    return wrapped

@app.route("/nodes/<string:label>", methods=["GET", "POST", "DELETE", "PUT"])
@is_admin
def nodes_view(label):
    node = session.query(Node).filter(Node.label==label).first()

    if request.method == "PUT":

        if node:
            
            if node.api_key == request.headers.get("X-API-Key",""):
                public_key = request.json["public_key"]
                if node.public_key:
                    peer_down(node.public_key)
                payload = add_peer(public_key, [node.allowed_ips for node in session.query(Node).all()])
                node.allowed_ips = payload["address"]
                node.public_key = public_key
                session.add(node)
                session.commit()
                return make_response(payload, 200)

            return make_403()

        return make_404()

    if request.method == "GET":

        if node:
            response = {"label": node.label}
            
            response["configured"] = (not config["WG_ON"]) or bool(node.allowed_ips)
            response["apps"] = [app.label for app in node.applications]

            if node.allowed_ips:
                response["address"] = node.allowed_ips
                response["public_key"] = node.public_key

            return make_response(response, 200)

        return make_404()

    if request.method == "POST":
        
        if node:
            return make_response("Conflicting names", 409)
        api_key = str(uuid4())
        node = Node(label=label, api_key=api_key)
        session.add(node)
        session.commit()
        return make_response(api_key, 200)

    if request.method == "DELETE":
        if node:
            if node.public_key:
                peer_down(node.public_key)
            session.delete(node)
            session.commit()
            return make_response("OK", 200)
        return make_404()

@app.route("/apps/<string:label>", methods=["GET", "POST", "DELETE"])
@is_admin
def apps_view(label):
    app = session.query(Application).filter(Application.label==label).first()

    if request.method == "GET":

        if app:
            response = {"label": app.label}

            policies = session.query(Policy).filter(Policy.application==app).all()
            response["nodes"] = {policy.node.label: policy.value for policy in policies}
            return make_response(response, 200)

        return make_404()

    if request.method == "POST":
        
        if app:
            return make_response("Conflicting names", 409)
        api_key = str(uuid4())
        app = Application(label=label, api_key=api_key)
        session.add(app)
        session.commit()
        return make_response(api_key, 200)

    if request.method == "DELETE":
        if app:
            session.delete(app)
            session.commit()
            return make_response("OK", 200)
        return make_404()

@app.route("/policies/<string:app_label>/<string:node_label>", methods=["GET", "POST", "DELETE"])
@is_admin
def policies_view(app_label, node_label):
    app = session.query(Application).filter(Application.label==app_label).first()
    node = session.query(Node).filter(Node.label==node_label).first()

    if not (app and node):
        return make_404()

    policy = session.query(Policy).filter(Policy.application==app,
                                          Policy.node==node).first()

    if request.method == "GET":

        if policy:
            return make_response(policy.value, 200)

        return make_404()

    if request.method == "POST":
        new_policy = request.json["policy"]

        if not policy:
            policy = Policy(value=new_policy)
            policy.node = node
            policy.application = app
        else:
            policy.value = new_policy
        session.add(policy)
        session.commit()
        return make_response(policy.value, 200)

    if request.method == "DELETE":
        if policy:
            session.delete(policy)
            session.commit()
            return make_response("OK", 200)
        return make_404()

@app.route("/jobs/<string:app_label>/<string:job_label>", methods=["GET", "POST", "DELETE", "PUT"])
def jobs_view(app_label, job_label):
    app = session.query(Application).filter(Application.label==app_label).first()
    
    if not app:
        return make_404()

    if app.api_key != request.headers.get("X-API-Key"):
        return make_403()

    if request.method == "GET":
        if app_label in jobs and job_label in jobs[app_label]:
            if "result" in jobs[app_label][job_label]:
                return jobs[app_label][job_label]["result"]
            try:
                status = jobs[app_label][job_label]["status_queue"].get_nowait()
            except Empty:
                status = None
            if status:
                jobs[app_label][job_label]["result"] = status
                return status
            else:
                return "Running"
        return make_404()

    if request.method == "POST":
        program = request.json["program"]
        if (app_label not in jobs) or (job_label not in jobs[app_label]):

            policies = session.query(Policy).filter(Policy.application==app).all()

            if not policies:
                return make_response("No policies found", 404)
    
            dpps = []
            base = config["IP_ADDRESS"] if config["WG_ON"] else config["HOST"]
            for model_id, policy in enumerate(policies):

                if config["WG_ON"]:
                    if policy.node.allowed_ips:
                        node_host = policy.node.allowed_ips
                        if "/" in node_host:
                            node_host = node_host.rstrip("0123456789").rstrip("/")
                    else:
                        continue
                else:
                    node_host = policy.node.label

                dpp = DataPolicyPair(policy=policy.value)
                dpp._data = {
                    "ip_address": node_host,
                    "callback_url": "http://{}/rpc/{}/{}/{}".format(base, app_label, job_label, policy.node.label),
                    "model_base_url": "http://{}/{}".format(base, config["WEBPATH"]),
                    "webroot": config["WEBROOT"],
                    "label": policy.node.label,
                    "model_id": model_id,
                    "timestamps": [], 
                }
                dpps.append(dpp)
            status_queue = Queue() 
            rpc_queue = Queue()
            thread = Thread(target=execute_program, args=(program, dpps, status_queue, rpc_queue, ))
            jobs.setdefault(app_label, {})
            jobs[app_label][job_label] = {
                "thread": thread,
                "status_queue": status_queue,
                "rpc_queue": rpc_queue
            }
            thread.start()

            return make_response("OK", 200)
        return make_response("Conflicting names", 409)
    if request.method == "DELETE":
        if job_label in jobs:
            jobs[job_label]["rpc_queue"].put(("", None, ))

@app.route("/rpc/<string:app_label>/<string:job_label>/<string:node_label>", methods=["POST"])
def rpc_view(app_label, job_label, node_label):
    node = session.query(Node).filter(Node.label==node_label,
                                         Node.api_key==request.headers.get("X-API-Key", "")).first()

    if not node:
        return make_403()

    if (app_label not in jobs) or (job_label not in jobs[app_label]):
        return make_404()

    queue = jobs[app_label][job_label]["rpc_queue"]
    json = request.json
    if json["status"] == "OK":
        if config["WG_ON"]:
            if node.allowed_ips:
                node_host = node.allowed_ips
                if "/" in node_host:
                    node_host = node_host.rstrip("0123456789").rstrip("/")
        else:
            node_host = policy.node.label

        queue.put_nowait((node_label, f"http://{node_host}/{json['data_policy_pair']}", ))
    else:
        queue.put_nowait((node_label, None, ))

    return make_response("OK", 200)

