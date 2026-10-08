import threading
import urllib.request

import pytest

from flyheal.ui import make_server


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
