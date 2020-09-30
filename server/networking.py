from ipaddress import IPv4Interface
from pyroute2 import IPDB, WireGuard
from server.config import config

IFNAME = config['IFNAME']
IFADDR = config['IFADDR']
WGPORT = config['WGPORT']

def start_wireguard(private_key, nodes):
    with IPDB() as ip:

        # connection already up
        if IFNAME in ip.interfaces:
            return

        wg_if = ip.create(kind='wireguard', ifname=IFNAME)
        wg_if.add_ip(IFADDR)
        wg_if.up()
        wg_if.commit()

    wg = WireGuard()
    wg.set(IFNAME, private_key=private_key, listen_port=WGPORT)
    
    for public_key, ip in nodes:
        peer_up(public_key, ip)

def stop_wireguard():
    
    with IPDB() as ip:
        ip.interfaces[IFNAME].remove().commit()

def wg_info():
    wg = WireGuard()
    info = wg.info(IFNAME)

    return {
        "public_key": info[0].WGDEVICE_A_PUBLIC_KEY['value'].decode(),
        "port": info[0].WGDEVICE_A_LISTEN_PORT['value']
    }

def get_taken_ips():
    
    wg = WireGuard()
    interface = IPv4Interface(IFADDR)

    taken_ips = {interface.network.network_address.compressed, interface.ip.compressed}
    for peer in wg.info(IFNAME)[0].WGDEVICE_A_PEERS.get('value', []):
        allowed_ips = peer.WGPEER_A_ALLOWEDIPS.get('value')
        if allowed_ips:
            taken_ips.add(IPv4Interface(allowed_ips[0]['addr']).ip.compressed)
    return taken_ips

def add_peer(public_key):

    network = IPv4Interface(IFADDR).network
    taken_ips = get_taken_ips()
    for ip in network:

        if str(ip) not in taken_ips:
            ip = f"{ip}/24"
            peer_up(public_key, ip)
            info = wg_info()
            return {
                "address": ip,
                "public_key": info["public_key"],
                "port": info["port"],
                "allowed_ips": network.compressed
            }

    raise Exception("Not enough IPs")

def peer_up(public_key, ipv4_address):
    peer = {
        "public_key": public_key,
        "allowed_ips": [ipv4_address]
    }

    wg = WireGuard()
    wg.set(IFNAME, peer=peer)


def peer_down(public_key):
    peer = {
        "public_key": public_key,
        "remove": True
    }

    wg = WireGuard()
    wg.set(IFNAME, peer=peer)

