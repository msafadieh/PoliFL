#! /bin/sh

umask 077

if ! [ -f "/data/privkey" ]; then
	wg genkey > /data/privkey
fi

umask 022
exec gunicorn -b 0.0.0.0:8000 server:app
