# PoliBox 

## Setting up the central node

### Requirements

- rabbitmq-server
- Python 3.7+
- NGINX
- OpenSSH

### Configuring OpenSSH

Make sure your server is accessible over SSH

### Configure RabbitMQ

Create a RabbitMQ user to be able to connect from outside localhost.

```
> sudo rabbitmqctl add_user test test
> sudo rabbitmqctl set_permissions -p / test ".*" ".*" ".*"
```

### Configure NGINX

Make sure that the user running the central has read-write accesst to `/var/www/html`.

### Configure Python environment

1. Clone this repository.

```
git clone https://github.com/minoskt/polibox.git
```

2. setup a virtual environment using `venv`.

```
> cd polibox
> python -m venv .env
> source .env/bin/activate
> pip install -r requirements.txt
```

3. Add users and policies to `users.txt` by using the template in `users_example.txt`. Each line has a username and password separated by a semi-colon.

```
username1;ANYF*
username2;ANYF*
...
```

## Setting up the edge node

### Requirements

- Python 3.7+
- OpenSSH

### Configuring OpenSSH

Setup ssh key-pair with no password and copy it over to the server

```
> ssh-keygen
Generating public/private rsa key pair.
Enter file in which to save the key (/home/mhmd/.ssh/id_rsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/mhmd/.ssh/id_rsa
Your public key has been saved in /home/mhmd/.ssh/id_rsa.pub
The key fingerprint is:
SHA256:KRdNDZ9xZgnqJUU/PUy0uxBaJ2C6k7ZFxyb1MseXAeY mhmd@eshmun
The key's randomart image is:
+---[RSA 3072]----+
|          o*==*= |
|         oo+BB*.+|
|        ..+o+EoXo|
|         ++o* B.+|
|      . S=.o . . |
|       o. +   . .|
|         .     . |
|                 |
|                 |
+----[SHA256]-----+

> ssh-copy-id -p 22 username@hostname
```

### Setup python environment

1. Clone this repository.

```
git clone https://github.com/minoskt/polibox.git
```

2. Setup a virtual environment using `venv`.

```
> cd PoliBox
> python -m venv .env
> source .env/bin/activate
> pip install -r requirements.txt
```

3. Create a configuration file `config.json` based on the provided `config_example.json`. The edge node's username is the same one that is associated witht the policy. Use the RabbitMQ credentials that you created on the central node.

```
{
	"USERNAME": "",
	"SERVER_URL": "",
	"SSH_USERNAME": "",
	"SSH_PORT": "",
	"RMQ_USERNAME": "",
	"RMQ_PASSWORD": ""
}
```

## Running the experiment

### On each edge node

To start an edge node, activate the python environment and run `edge.py`.

```
> cd PoliBox
> source .env/bin/activate
> python edge.py
```

### On the central node

Modify `federated.select_users`, `general.sample_data_policy_pairs`, and `federated.average` in `program.py` tp match the number of users in `users.txt`. Activate the python environment and run `central.py`.

```
> cd PoliBox
> source .env/bin/activate
> python central.py
```
