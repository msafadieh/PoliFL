import os
import dill
import pika
import json

def send_message(target, body, corr_id, channel, callback, create=False):
    
    with open('config_central.json') as f:
        config = json.load(f)
    
    url = config["URL"]
    port = config["PORT"]

    callback_queue = channel.queue_declare(queue=f'{target}_reply', durable=True).method.queue
    channel.basic_consume(
        queue=callback_queue,
        on_message_callback=callback,
        auto_ack=True)
    filename = f'/var/www/html/{corr_id}'
    with open(filename, 'wb') as f:
        dill.dump(body, f)

    channel.basic_publish(
            exchange='',
            routing_key=target,
            properties=pika.BasicProperties(
                correlation_id=corr_id #properties.correlation_id,
            ),
            body=f"{url}:{port}/model")

