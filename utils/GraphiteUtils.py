import pickle
import socket
import struct
from typing import List, Tuple


def send_time_data(database_path: str, data: List[dict]):
    content: List[Tuple[str, Tuple[str, str]]] = []
    for value in data:
        content.append((database_path, (value.get("timestamp"), value.get("value"))))
    payload = pickle.dumps(content, protocol=2)
    header = struct.pack("!L", len(payload))
    message = header + payload
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("127.0.0.1", 2004))
    client_socket.send(message)
