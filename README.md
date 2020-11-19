# PoliBox

## Terms used

- **Central service**: Refers to the main installation of ancile, as defined in `Dockerfile.server`
- **Edge node**: Refers to each individual node, defined in `Dockerfile.client`
- **Corpus**: Refers to [this file](https://drive.google.com/file/d/1qTfiZP4g2ZPS5zlxU51G-GDCGGr23nvt/view?usp=sharing) (or a smaller subset of it)

### Requirements

Requirements are the same for both the central service and edge nodes.

- Docker
- WireGuard (if using Linux 5.9+, then it's already included in the kernel)

## Deploying central service

1. Generate a password to be used with the central service's admin API.
2. Clone this repo and `cd` into it.
3. Create directory called `data` and save the corpus as `data/corpus.pt.tar`. 
4. Build the docker image: `docker build -t ancile-server -f Dockerfile.server .`
5. Run it using the following command, make sure to replace `$API_KEY` with the key generated in step 1:
```
docker run \
	--name="ancile-server" \
	--cap-add=NET_ADMIN \
	--cap-add=SYS_MODULE \
	-p 59000:59000/udp \
	-p 80:80/tcp \
	-v /lib/modules:/lib/modules \
	-v data:/data \
	-e API_KEY="$API_KEY" \
	ancile-server:latest
```

Your server should be listening on port 80.

## Deploying edge node

1. Make a POST request to your central service at `/nodes/$NODENAME` with the `X-API-Key` header set to the admin API key you generated and with a unique label for your edge node instead of `$NODENAME`. You should receive a token for your node as a response.
2. Clone this repo and `cd` into it.
3. Build the docker image: `docker build -t ancile-client -f Dockerfile.client .`
4. Run it using the following command. Make sure to replace `$NODE_NAME`, `$NODE_KEY`, and `http://my-central-service` with correct values:
```
docker run \
	--name="ancile-client" \
	--cap-add=NET_ADMIN \
	--cap-add=SYS_MODULE \
	-v /lib/modules:/lib/modules \
	-e NODE_NAME=$NODE_NAME \
	-e NODE_KEY=$NODE_KEY \
	-e SERVER_ENDPOINT=http://my-central-service \
	ancile-client:latest
```

Your edge node just establish a secure VPN connection with the central service.

## Running experiment

1. Create application by sending POST request to `/apps/$APPNAME` with the same header you used in step 1 of deploying the edge node and replacing `$APPNAME` with your app label. You will receive an app key back.
2. Create a policy for each node with your app by making a POST request to `/policies/$APPNAME/$NODENAME` with your admin key in the header, and your policy in the body as a JSON in the following format: `{"policy": "$MYPOLICY"}`.
3. Create a job by making a POST request to `/jobs/$APPNAME/$JOBNAME` with `X-API-Key` header set to your app's key that you received in step 1, and your program in the body as a JSON in the following format: `{"program": "$MYPROGRAM"}`. *You can also use start.py from the repo which uses the content of program.py*.
4. You can check the status of your job by making a GET request to `/jobs/$APPNAME/$JOBNAME` with your app key in the header.
