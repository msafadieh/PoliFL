from ancile.core.decorators import *
from ancile.lib.general import get_token
from ancile.utils.errors import AncileException

name="databox"

@ExternalDecorator()
def get_latest_reddit_data(user, session):
    import json
    import random
    import requests

    resp = json.loads(requests.get("https://gist.githubusercontent.com/msafadieh/869c51a75bc77e5d8746b432c7a1f354/raw/6cfd0f7dacc317ba4dd58831014fdfb6dc2b3f66/data.json").text)
    
    sents = json.loads(resp['data'][0]['data']['data'])
    return random.choice(sents)

    url = "https://127.0.0.1/app-ancile/ui/tsblob/latest"
    payload = { "data_source_id": "redditSimulatorData"}
    headers = { "session": session }
    res = requests.get(url, cookies=headers, params=payload, verify=False)
    if res.status_code == 200:
        data = res.json()
    else:
        raise AncileException("Couldn't fetch data from databox.")
    string = data['data'][0]['data']['data']
    print(string, flush=True)
    return string
