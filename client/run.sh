#! /bin/sh

set -e

WG_PATH=${WG_PATH:-/etc/wireguard}
WG_IFNAME=${WG_IFNAME:-wg0}

CONFIG_PATH=${CONFIG_PATH:-/etc/wireguard/$WG_IFNAME.conf}

if [ -z "${SERVER_ENDPOINT:-}" ]; then
	echo "SERVER_ENDPOINT not found" >&2
	exit 1
fi

if ! [ -f "${CONFIG_PATH}" ]; then

	if [ -z "${NODE_NAME:-}" ]; then
		echo "NODE_NAME not found" >&2
		exit 1
	elif [ -z "${NODE_KEY:-}" ]; then
		echo "NODE_KEY not found" >&2
		exit 1
	fi
	
	export NODE_NAME
	export NODE_KEY
	export CONFIG_PATH
	PRIVATE_KEY="$(wg genkey)" PUBLIC_KEY="$(echo "$PRIVATE_KEY" | wg pubkey)" run.py
fi

wg-quick up "$WG_IFNAME"

wg
exec gunicorn -b 0.0.0.0:8000 client:app
