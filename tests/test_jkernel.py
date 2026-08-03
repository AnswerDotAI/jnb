"The J kernel through the full Jupyter protocol: results, state, definitions, errors, interrupt - driven over real sockets by kernmini's MiniSession."

import json, os, socket, subprocess, sys, time

import pytest, zmq

from kernmini.session import MiniSession



def _sock(ctx, typ, port):
    s = ctx.socket(typ)
    s.linger = 0
    if typ == zmq.SUB: s.setsockopt(zmq.SUBSCRIBE, b"")
    s.connect(f"tcp://127.0.0.1:{port}")
    return s


def _ports(n):
    socks = [socket.socket() for _ in range(n)]
    for s in socks: s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks: s.close()
    return ports


def _drain_iopub(sub, until_idle=True, timeout=10.0):
    "Collect iopub msg dicts until an idle status (skipping the welcome)."
    sess, out = _drain_iopub.sess, []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not sub.poll(200): continue
        frames = sub.recv_multipart()
        idents, rest = sess.feed_identities(frames)
        msg = sess.deserialize(rest)
        if msg["msg_type"] == "iopub_welcome": continue
        out.append(msg)
        if until_idle and msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle": return out
    raise TimeoutError(f"no idle within {timeout}s; got {[m['msg_type'] for m in out]}")


def _await_welcome(sub, timeout=60.0):
    "Wait for the JEP 65 iopub_welcome: proof the subscription is live, so no later message can be missed."
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not sub.poll(200): continue
        _, rest = _drain_iopub.sess.feed_identities(sub.recv_multipart())
        if _drain_iopub.sess.deserialize(rest)["msg_type"] == "iopub_welcome": return
    raise TimeoutError("no iopub_welcome")


def _request(sock, sess, msg_type, content, timeout=10.0):
    sock.send_multipart(sess.serialize(sess.msg(msg_type, content)))
    if not sock.poll(timeout * 1000): raise TimeoutError(f"no reply to {msg_type}")
    idents, rest = sess.feed_identities(sock.recv_multipart())
    return sess.deserialize(rest)


@pytest.fixture
def j_kernel(tmp_path):
    key = "test-key-123"
    shell_p, iopub_p, stdin_p, control_p, hb_p = _ports(5)
    conn = dict(transport="tcp", ip="127.0.0.1", shell_port=shell_p, iopub_port=iopub_p, stdin_port=stdin_p,
        control_port=control_p, hb_port=hb_p, key=key, signature_scheme="hmac-sha256")
    cf = tmp_path / "conn.json"
    cf.write_text(json.dumps(conn))
    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")  # isolate from any real startup.ijs
    proc = subprocess.Popen([sys.executable, "-m", "jnb.jkernel", "-f", str(cf)], stderr=subprocess.PIPE, env=env)
    ctx = zmq.Context.instance()
    sess = MiniSession(key=key.encode(), username="testclient")
    _drain_iopub.sess = MiniSession(key=key.encode())
    shell, control, sub = _sock(ctx, zmq.DEALER, shell_p), _sock(ctx, zmq.DEALER, control_p), _sock(ctx, zmq.SUB, iopub_p)
    _await_welcome(sub)
    try: yield proc, sess, shell, control, sub
    finally:
        for s in (shell, control, sub): s.close(0)
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_jkernel_end_to_end(j_kernel):
    "Results, state, multiline definitions, errors, and interrupt through the whole protocol."
    proc, sess, shell, control, sub = j_kernel

    info = _request(shell, sess, "kernel_info_request", {}, timeout=60)  # first reply waits for the J boot
    c = info["content"]
    assert c["implementation"] == "jkernel" and c["language_info"]["name"] == "J"
    assert c["banner"].startswith("J j") and c["supported_features"] == []
    _drain_iopub(sub)

    def run(code, timeout=10.0):
        reply = _request(shell, sess, "execute_request", dict(code=code), timeout=timeout)
        msgs = _drain_iopub(sub)
        out = "".join(m["content"]["text"] for m in msgs if m["msg_type"] == "stream")
        return reply["content"], out

    reply, out = run("2+2")
    assert reply["status"] == "ok" and out == "4\n"
    reply, out = run("x =: 41")
    assert reply["status"] == "ok" and out == ""                      # assignment is silent
    assert run("x + 1")[1] == "42\n"                                  # state persists
    assert run("mean =: 3 : 0\n(+/ y) % # y\n)\nmean 1 2 3 4")[1] == "2.5\n"

    reply, out = run("1 + 'a'")
    assert reply["status"] == "error" and reply["ename"] == "JError" and "domain error" in reply["evalue"]
    assert run("2+2")[1] == "4\n"                                     # session survived the error

    run("spin =: 3 : 0\nn =. 0\nwhile. n < 1e9 do. n =. n + 1 end.\n)")
    shell.send_multipart(sess.serialize(sess.msg("execute_request", dict(code="spin 0"))))
    time.sleep(0.5)                                                   # let it start spinning...
    reply = _request(control, sess, "interrupt_request", {})          # ...then a J attention interrupt
    assert reply["content"]["status"] == "ok"
    assert shell.poll(10_000), "interrupted execute never replied"
    _, rest = sess.feed_identities(shell.recv_multipart())
    reply = sess.deserialize(rest)
    assert reply["content"]["status"] == "error" and "attention interrupt" in reply["content"]["evalue"]
    _drain_iopub(sub)
    assert run("2+2")[1] == "4\n"                                     # workspace survives an interrupt

    reply = _request(control, sess, "shutdown_request", dict(restart=False))
    assert reply["content"]["status"] == "ok"
    assert proc.wait(timeout=10) in (0, -9)


def test_startup_ijs(tmp_path):
    "startup.ijs runs in the session before the first request; its state persists."
    xdg = tmp_path / "xdg" / "jnb"
    xdg.mkdir(parents=True)
    (xdg / "startup.ijs").write_text("greeting =: 'hi from startup'\n")

    key = "test-key-123"
    shell_p, iopub_p, stdin_p, control_p, hb_p = _ports(5)
    conn = dict(transport="tcp", ip="127.0.0.1", shell_port=shell_p, iopub_port=iopub_p, stdin_port=stdin_p,
        control_port=control_p, hb_port=hb_p, key=key, signature_scheme="hmac-sha256")
    cf = tmp_path / "conn.json"
    cf.write_text(json.dumps(conn))
    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")
    proc = subprocess.Popen([sys.executable, "-m", "jnb.jkernel", "-f", str(cf)], stderr=subprocess.PIPE, env=env)
    ctx = zmq.Context.instance()
    sess = MiniSession(key=key.encode(), username="testclient")
    _drain_iopub.sess = MiniSession(key=key.encode())
    shell, sub = _sock(ctx, zmq.DEALER, shell_p), _sock(ctx, zmq.SUB, iopub_p)
    try:
        _await_welcome(sub)
        reply = _request(shell, sess, "execute_request", dict(code="greeting"), timeout=60)
        msgs = _drain_iopub(sub)
        out = "".join(m["content"]["text"] for m in msgs if m["msg_type"] == "stream")
        assert out == "hi from startup\n"
    finally:
        for s in (shell, sub): s.close(0)
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
