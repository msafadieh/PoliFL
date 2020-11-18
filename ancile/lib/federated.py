from ancile.core.decorators import TransformDecorator
name = 'federated'

def new_model(policy):
    from ancile.core.primitives import DataPolicyPair
    import yaml
    from ancile.utils.text_load import load_data
    from ancile.lib.federated_helpers.utils.text_helper import TextHelper

    corpus = load_data('/data/corpus.pt.tar')
    with open('ancile/lib/federated_helpers/utils/words.yaml') as f:
        params = yaml.load(f)
    helper = TextHelper(params=params, current_time='None',
                        name='databox', n_tokens=50000)
    helper.load_data(corpus=corpus)
    model = helper.create_one_model().state_dict()
    dpp = DataPolicyPair(policy=policy)
    dpp._data = {"model": model, "helper": helper}
    return dpp

def select_users(user_count, dpps):
    import random
#    from ancile.core.primitives import DataPolicyPair

    if len(dpps) < user_count:
        raise Exception("Not enough users")
    
    return random.sample(dpps, user_count)

#    with open('config/users.txt') as f:
#        user_policy = [u.split(";") for u in f.read().split('\n') if u]
#
#    return dpps
#
#    sample = random.sample(user_policy, user_count)
#    for model_id, target in enumerate(sample):
#        
#        target_name, policy = target
#        dpp = DataPolicyPair(policy=policy)
#        dpp._data = {
#            "target_name": target_name,
#            "model_id": model_id
#        }
#
#        dpps.append(dpp)


class RemoteClient:
    def __init__(self, callback, queue):
        self.callback = callback
        self.callback_result = None
        self.queue = queue
        self.nodes = set()
        self.error = None

    def __process_model(self, model_url):
        from io import BytesIO
        import pycurl
        import dill
        bytes_buffer = BytesIO() 
        curl = pycurl.Curl()
        curl.setopt(curl.URL, model_url)
        curl.setopt(curl.WRITEDATA, str_buffer)
        curl.setopt(curl.CAINFO, certifi.where())
        curl.perform()
        curl.close()

        dpp = dill.load(bytes_buffer)
        self.callback_result = self.callback(initial=self.callback_result, dpp=dpp)

    def send_to_edge(self, model, participant_dpp, program):
        from ancile.core.primitives import DataPolicyPair
        import dill
        import uuid
        import os
        import requests

        dpp_to_send = DataPolicyPair(policy=participant_dpp._policy)
        dpp_to_send._data = {
                "global_model": model._data["model"],
                "helper": model._data["helper"],
                "model_id": participant_dpp._data["model_id"],
                }

        ip_address = participant_dpp._data["ip_address"]
        callback_url = participant_dpp._data["callback_url"]
        model_base_url = participant_dpp._data["model_base_url"]
        webroot = participant_dpp._data["webroot"]
        label = participant_dpp._data["label"]       
        self.nodes.add(label)

        uuid = str(uuid.uuid4())
        with open(os.path.join(webroot, uuid), 'wb+') as f:
            dill.dump(dpp_to_send, f)

        body = {
                "program": program,
                "model_url": f"{model_base_url}/{uuid}",
                "callback_url": callback_url
        }

        url = f"http://{ip_address}/execute"
        print(f"queuing {label}: {url}")
        resp = requests.post(url, json=body)
        print(resp.text)

    def poll_and_process_responses(self):
        while (not self.error) and self.nodes:
            node, model_url = self.queue.get()
            if self.error:
                continue
            
            if not node:
                self.error = self.error or "Execution cancelled"
            elif not model_url:
                self.error = self.error or "error on node: {}".format(node)
            else:
                self.__process_model(model_url)
                self.nodes.remove(node)

        if self.error:
            raise Exception(self.error)

        return self.callback_result

@TransformDecorator()
def train_local(model, data_point):
    """
    This part simulates the

    """
    from ancile.lib.federated_helpers.training import _train_local
    model["train_data"] = data_point
    output = _train_local(**model)
    return output


@TransformDecorator()
#@profile
def accumulate(initial, dpp):
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled= True
    torch.backends.cudnn.benchmark= True
    # averaging part
    initial = initial or dict()
    counter = initial.pop("counter", 0)
    initial = initial.pop("initial", dict())
    for name, data in dpp.items():
        #### don't scale tied weights:
        if name == 'decoder.weight' or '__' in name:
            continue
        if initial.get(name, False) is False:
            initial[name] = torch.zeros_like(data, requires_grad=True)
        with torch.no_grad():
            initial[name].add_(data)
        del data
    initial["counter"] = counter+1
    initial["initial"] = initial
    return initial


@TransformDecorator()
def average(accumulated, model, enforce_user_count=0): #summed_dps, global_model, eta, diff_privacy=None, enforce_user_count=0):
    import torch

    eta = 100
    diff_privacy = None
    helper = model["helper"]
    model = model["model"]
    accumulated = accumulated or dict()
    if enforce_user_count and enforce_user_count > accumulated.get("counter", 0):
        raise Exception("User count mismatch")

    accumulated = accumulated.get("initial", {})

    for name, data in model.items():
        #### don't scale tied weights:
        if name == 'decoder.weight' or '__' in name:
            continue

        update_per_layer = accumulated[name] * eta
        if diff_privacy:
            noised_layer = torch.cuda.FloatTensor(data.shape).normal_(mean=0, std=diff_privacy['sigma'])
            update_per_layer.add_(noised_layer)
        with torch.no_grad():
            data.add_(update_per_layer)
    return model
