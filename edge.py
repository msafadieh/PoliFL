import os
import subprocess
import dill
import pika
import requests
import torch
import ancile
from ancile.core.core import execute

def callback(edge_username, host, port, username)
    def debug(msg):
        print(f"[{edge_username}]: {msg}")
    
    def func(ch, method, properties, body):
        debug(f"Received message of length {len(body)}")
        request = requests.get(body)
        dpp = request.get("data_policy_pair")
        program = request.get("program")
        if not dpp:
            message = {"error": "data_policy_pair missing"}
        elif not program:
            message = {"error": "program missing"}
        else:
            res = execute(users_secrets=[],
                           program=program,
                            app_id=None,
                            app_module=None,
                            data_policy_pairs=[dpp])
            if res["result"] == "error":
                message = {"error": res["traceback"]}
            elif not res["data"]:
                message = {"error": "no dpp found"}
            else:
                message = {"data_policy_pair": dpp}

        filename = f'/tmp/{properties.correlation_id}'
        with open(filename, 'wb') as f:
            dill.dump(message, f)
        print(f"Transferring to host")
        subprocess.run(["scp", "-P", port, filename, f"{username}@{host}:{filename}"]) 
        os.remove(filename)
        debug(f"Returning response: {times}")
        channel.basic_publish(
            exchange='',
            routing_key=host+'_reply',
            properties=pika.BasicProperties(
                correlation_id=properties.correlation_id,
            ),
            body='')

    return func

def main():
    with open("config/config_edge.json") as f:
        configs = json.load(f)

    username = configs.get("USERNAME")

    ssh_port = configs.get("SSH_PORT")
    ssh_username = configs.get("SSH_USERNAME")
    host = configs.get("SERVER_URL")
    rmq_username = configs.get("RMQ_USERNAME")
    rmq_password = configs.get("RMQ_PASSWORD")

    callback_fn = callback(username, host, ssh_port, ssh_username) 

    creds = pika.PlainCredentials(rmq_username, rmq_password)
    params = pika.ConnectionParameters(host=host, credentials=creds)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=username)
    channel.basic_consume(queue=username, on_message_callback=callback, auto_ack=True)
    print(f'[{username}] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == "__main__":
    main()
