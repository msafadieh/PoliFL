docker run \
	--name="ancile-server" \
	--cap-add=NET_ADMIN \
	--cap-add=SYS_MODULE \
	-p 59000:59000/udp \
	-v /lib/modules:/lib/modules \
	-e API_KEY "$1" \
	ancile-server:latest

