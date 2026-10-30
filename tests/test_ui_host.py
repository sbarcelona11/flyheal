import http.client
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

from flyheal.ui import make_server, parse_allowed_hosts


def _post(port, path, headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    try:
        connection.request('POST', path, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


@pytest.fixture
def server(tmp_path):
    instance = make_server(tmp_path / 'missing', port=0)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize('host', ['127.0.0.1', '0.0.0.0'])
def test_server_serves_web_on_requested_interface(tmp_path, host):
    server = make_server(tmp_path / 'missing', port=0, host=host)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.server_address[0] == host
        with urllib.request.urlopen(
            f'http://127.0.0.1:{server.server_port}/'
        ) as response:
            assert response.status == 200
            assert b'FlyHeal' in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_origin_is_accepted(server):
    port = server.server_port
    status, _ = _post(port, '/api/cancel',
                      headers={'Origin': f'http://127.0.0.1:{port}'})
    assert status == 200


def test_proxy_https_origin_is_accepted(server):
    status, _ = _post(server.server_port, '/api/cancel',
                      headers={'Host': 'flyheal.onrender.com',
                               'Origin': 'https://flyheal.onrender.com'})
    assert status == 200


def test_x_forwarded_host_matching_origin_is_accepted(server):
    status, _ = _post(server.server_port, '/api/cancel',
                      headers={'Host': '127.0.0.1:9999',
                               'X-Forwarded-Host': 'flyheal.onrender.com',
                               'Origin': 'https://flyheal.onrender.com'})
    assert status == 200


def test_foreign_origin_is_rejected(server):
    status, _ = _post(server.server_port, '/api/cancel',
                      headers={'Host': 'flyheal.onrender.com',
                               'Origin': 'https://otro-sitio.com'})
    assert status == 403


def test_missing_origin_and_referer_is_rejected(server):
    status, _ = _post(server.server_port, '/api/cancel', headers={})
    assert status == 403


def test_allowed_hosts_blocks_dns_rebinding(tmp_path):
    server = make_server(tmp_path / 'missing', port=0, allowed_hosts={'localhost'})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _ = _post(server.server_port, '/api/cancel',
                          headers={'Host': 'evil.com', 'Origin': 'http://evil.com'})
        assert status == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_allowed_hosts_accepts_listed_host(tmp_path):
    server = make_server(tmp_path / 'missing', port=0,
                         allowed_hosts={'flyheal.onrender.com'})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _ = _post(server.server_port, '/api/cancel',
                          headers={'Host': 'flyheal.onrender.com',
                                   'Origin': 'https://flyheal.onrender.com'})
        assert status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_parse_allowed_hosts_splits_and_normalizes():
    assert parse_allowed_hosts('flyheal.onrender.com,localhost,127.0.0.1') == \
        {'flyheal.onrender.com', 'localhost', '127.0.0.1'}
    assert parse_allowed_hosts('  LocalHost , 127.0.0.1 ') == {'localhost', '127.0.0.1'}


def test_port_env_makes_server_listen_on_that_port(tmp_path):
    env = os.environ.copy()
    env['PORT'] = '9999'
    process = subprocess.Popen(
        [sys.executable, '-m', 'flyheal.ui', '--results', str(tmp_path / 'missing'), '--no-open'],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                sock.settimeout(0.1)
                if sock.connect_ex(('127.0.0.1', 9999)) == 0:
                    break
            time.sleep(0.1)
        else:
            raise AssertionError('el servidor no escucha en 9999')
    finally:
        process.terminate()
        process.wait(timeout=5)
