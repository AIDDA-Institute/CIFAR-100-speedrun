"""Translate process cancellation into cleanup and preserved run results."""

import signal
import threading
from contextlib import contextmanager


class RunInterrupted(KeyboardInterrupt):
    def __init__(self, signum):
        super().__init__(f"Received {signal.Signals(signum).name}")
        self.signum = signum
        self.result_dir = None
        self.summary = None


@contextmanager
def handle_interrupts():
    # Python only permits signal handlers in the main thread. A caller running
    # this function in a thread remains responsible for process-level signals.
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    received = False

    def interrupt(signum, frame):
        nonlocal received
        if not received:
            received = True
            raise RunInterrupted(signum)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, interrupt)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
