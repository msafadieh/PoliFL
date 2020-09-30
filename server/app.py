from functools import wraps
from uuid import uuid4
from flask import Flask, make_response, request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.config import config
from server.models import Application, Base, Node, Policy
from server.networking import start_wireguard, add_peer, peer_down

engine = create_engine(config['SQLALCHEMY_DATABASE_URI'])
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

queues = {}

app = Flask(__name__)

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
                payload = add_peer(public_key)
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
            
            response["configured"] = bool(node.allowed_ips)
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

@app.route("/jobs/<string:app_label>/<string:job_label>/", methods=["GET", "POST", "DELETE", "PUT"])
def jobs_view(app_label, job_label):
    app = session.query(Application).filter(Application.label==app_label).first()
    
    if not app:
        return make_404()

    if request.method == "GET":
        if app_label in jobs and job_label in jobs[app_label]:
            status = status_queue.get_nowait()
            if status:
                jobs[app_label].pop(job_label)
                return status
            else:
                return "Running"
        return make_404()

    if request.method == "POST":
        if (app_label not in jobs) or (job_label not in jobs[app_label]):

            policies = session.query(Policy).filter(Policy.application==app).all()

            if not policies:
                return make_response("No policies found", 404)
    
            dpps = []
            for policy in policies:
                dpp = DataPolicyPair(policy=policy.value)
                dpp._data = {
                    "ip_address": config["IP_ADDRESS"],
                    "callback_url": "http://{}/rpc/{}/{}/{}".format(config["IP_ADDRESS"], app_label, job_label policy.node.label),
                    "model_base_url": "http://{}/{}".format(config["IP_ADDRESS"], config["WEBPATH"]),
                    "webroot": config["WEBROOT"],
                    "label": policy.node.label,
                }
                dpps.append(dpp)
            status_queue = Queue() 
            rpc_queue = Queue()
            jobs[app_label] = jobs[app_label] or {}
            jobs[app_label][job_label] = {
                "thread": Thread(target=execute_program, args=(dpps, status_queue, rpc_queue, )),
                "status_queue": status_queue,
                "rpc_queue": rpc_queue
            }

            return make_response(200, "OK")

    if request.method == "DELETE":
        if job_label in jobs:
            jobs[job_label]["rpc_queue"].put(("", None, ))

@app.route("/rpc/<string:app_label>/<string:job_label>/<string:node_label>")
def rpc_view(app_label, job_label, node_label):
    node = db.session.query(Node).filter(Node.node_label=node_label,
                                         Node.api_key=request.headers.get("X-API-Key", ""))

    if not node:
        return make_403()

    if (app_label not in jobs) or (job_label not in jobs[app_label]):
        return make_404()

    queue = jobs[app_label][job_label]["rpc_queue"]
    json = request.json
    if json["status"] == "OK":
        queue.put_nowait((node_label, json["data_policy_pair"], ))
    else:
        queue.put_nowait((node_label, None, ))

    return make_response("OK", 200)

