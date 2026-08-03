"The J Jupyter kernel: an in-process libj session behind kernmini. Hand-written (not nbdev-exported)."

import argparse, asyncio, logging, sys
from contextlib import contextmanager

from fastcore.xdg import xdg_config_home
from kernmini import install_kernelspec, run_kernel

from .j import J, JError

log = logging.getLogger("jnb.jkernel")


class JShell:
    "kernmini's shell contract over a libj session: J output streams as the session transcript."

    def __init__(self, request_input=None, **kw):
        self.j = J()
        self.execution_count = 0
        self._stream = None
        startup = xdg_config_home() / "jnb" / "startup.ijs"
        if startup.exists():
            try: self.j.run(startup.read_text())
            except Exception: log.warning("startup.ijs failed", exc_info=True)

    def set_stream_sender(self, sender): self._stream = sender

    @contextmanager
    def execution_context(self, allow_stdin, silent): yield

    async def execute(self, code, silent=False, store_history=True, user_expressions=None, allow_stdin=False):
        "Run J code on a worker thread (JDo blocks its caller at the C level), so the subshell loop stays free for interrupts."
        self.execution_count += 1
        try: out = await asyncio.get_running_loop().run_in_executor(None, self.j.run, code)
        except JError as e:
            err = str(e).rstrip()
            return dict(execution_count=self.execution_count, error=dict(ename="JError", evalue=err, traceback=[err]))
        if out and self._stream and not silent: self._stream("stdout", out)
        return dict(execution_count=self.execution_count)

    def interrupt(self):
        "A J attention interrupt: write the break flag (safe cross-thread); the engine polls it between sentences, ending the in-flight JDo."
        self.j.interrupt()
        return True

    def kernel_info(self):
        try: banner = self.j.run("9!:14 $0").strip()
        except Exception: banner = "unknown"
        version = banner.split("/")[0].lstrip("j")
        return dict(implementation="jkernel", implementation_version="0.1.0", banner=f"J {banner}",
            language_info=dict(name="J", version=version, mimetype="text/plain", file_extension=".ijs"))


def main():
    "CLI entry: `python -m jnb.jkernel -f <connection_file>`, or `install` for the kernelspec."
    if sys.argv[1:2] == ["install"]:
        dest = install_kernelspec("jkernel", [sys.executable, "-m", "jnb.jkernel", "-f", "{connection_file}"],
            display_name="J (jnb)", language="J")
        print(f"installed kernelspec: {dest}")
        return
    parser = argparse.ArgumentParser(prog="jkernel")
    parser.add_argument("-f", "--connection-file", required=True)
    args = parser.parse_args()
    run_kernel(args.connection_file, JShell, subshells=False)


if __name__ == "__main__": main()
