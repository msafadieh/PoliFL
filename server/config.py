from os import environ
import sys

config = {
    "API_KEY": environ["API_KEY"],
    "IFADDR": environ.get("IFADDR", "10.253.0.1/24"),
    "IFNAME": environ.get("IFNAME", "ancile"),
    "PRIVKEY_PATH": environ.get("PRIVKEY_PATH", "/data/privkey"),
    "WGPORT": int(environ.get("WGPORT", 59000)),
    "SQLALCHEMY_DATABASE_URI": environ.get("SQLALCHEMY_DATABASE_URI", "sqlite:////data/ancile.db"),
}
