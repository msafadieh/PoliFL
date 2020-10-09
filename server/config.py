from os import environ
import socket
import sys

config = {
    "WG_ON": environ.get("WG_ON", "1") != "0",
    "API_KEY": environ["API_KEY"],
    "WEBPATH": environ["WEBPATH"],
    "WEBROOT": environ["WEBROOT"],
    "HOST": environ.get("HOST", socket.gethostname()),
    "IFADDR": environ.get("IFADDR", "10.253.0.1/24"),
    "IP_ADDRESS": environ.get("IP_ADDRESS", "10.253.0.1"),
    "IFNAME": environ.get("IFNAME", "ancile"),
    "PRIVKEY_PATH": environ.get("PRIVKEY_PATH", "/data/privkey"),
    "WGPORT": int(environ.get("WGPORT", 59000)),
    "SQLALCHEMY_DATABASE_URI": environ.get("SQLALCHEMY_DATABASE_URI", "sqlite:////data/ancile.db"),
}
