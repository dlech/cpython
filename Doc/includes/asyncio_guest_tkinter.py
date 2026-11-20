#!/usr/bin/env python3
"""Minimal demo: asyncio running as a guest inside Tkinter's mainloop.

A progress bar counts from 0 to MAX_COUNT using ``asyncio.sleep()``.
The Tk GUI stays fully responsive throughout.  Closing the window or
pressing the Cancel button cancels the async task cleanly.

Usage::

    python asyncio_guest_tkinter.py
"""

import asyncio
import collections
import tkinter as tk
import tkinter.ttk as ttk
import traceback


# -- Host adapter for Tkinter ------------------------------------------

class TkHost:
    """Bridge between asyncio guest mode and the Tk event loop.

    Every tkinter call made from another thread waits for Tk's main
    loop, so this posts with Tk's ``after idle``, through ``root.call()``,
    which waits too.  That is thread-safe with a threaded Tcl, the
    default, but called from another thread, it raises RuntimeError if
    the thread that created the root isn't in ``mainloop()`` within
    about a second, which abandons the run.  So the host runs
    ``root.mainloop()``, not ``update()`` in a loop, until done_callback.

    ``after idle``, not ``after 0``: Tk redraws widgets in idle handlers,
    which run only when no other events are pending, so a busy guest
    posting its next batch with ``after 0`` would starve them, while
    ``after idle`` takes turns with them.  Guest code must not call
    ``update()`` or ``update_idletasks()``: they could run a callable the
    run has already posted while the batch is still running.

    From another thread, ``root.call()`` does not return until Tk's main
    loop has run the ``after idle`` command, though not the callable.  So
    posting from the I/O thread blocks it while the main thread is busy,
    and if the root were destroyed while such a call is waiting, the I/O
    thread would wait forever and the interpreter would never exit.
    Hence the root is destroyed only in done_callback, after the last
    post.  On Unix, ``root.tk.createfilehandler()`` avoids blocking: post
    by appending to the queue and sending a byte, without blocking, to
    one end of a socket.socketpair().  The file handler on the other end
    reads the bytes and posts the queue's runner with ``after idle``,
    rather than running it, so that a busy guest doesn't starve redraws.
    """

    def __init__(self, root):
        self.root = root
        self._tk_func_name = root.register(self._dispatch)
        self._q = collections.deque()

    def _dispatch(self):
        self._q.popleft()()

    def run_sync_soon_threadsafe(self, fn):
        """Schedule *fn* on the Tk thread."""
        self._q.append(fn)
        self.root.call('after', 'idle', self._tk_func_name)

    def done_callback(self, task):
        """Called when the async task finishes."""
        if task.cancelled():
            print("Task was cancelled.")
        elif task.exception() is not None:
            exc = task.exception()
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        else:
            print(f"Task returned: {task.result()}")
        self.root.destroy()


# -- Async workload ----------------------------------------------------

MAX_COUNT = 20
PERIOD = 0.5  # seconds between increments


async def count(progress, root):
    """Increment a progress bar, updating the Tk GUI each step."""
    root.wm_title(f"Counting every {PERIOD}s ...")
    progress.configure(maximum=MAX_COUNT)

    task = asyncio.current_task()
    loop = asyncio.get_running_loop()

    # Wire the Cancel button and window close to task.cancel().
    # Use call_soon_threadsafe so the I/O thread's selector is woken.
    def request_cancel():
        loop.call_soon_threadsafe(task.cancel)

    def on_close():
        # Not root.destroy(): the run needs Tk's main loop until
        # done_callback.  (If the run is abandoned after a failure,
        # done_callback never comes, so mainloop() never returns, and
        # the process keeps running with no window.)
        root.withdraw()
        request_cancel()

    cancel_btn = root.nametowidget('cancel')
    cancel_btn.configure(command=request_cancel)
    root.protocol("WM_DELETE_WINDOW", on_close)

    for i in range(1, MAX_COUNT + 1):
        await asyncio.sleep(PERIOD)
        progress.step(1)
        root.wm_title(f"Count: {i}/{MAX_COUNT}")

    return i


# -- Main ---------------------------------------------------------------

def main():
    root = tk.Tk()
    root.wm_title("asyncio guest + Tkinter")

    progress = ttk.Progressbar(root, length='6i')
    progress.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

    cancel_btn = tk.Button(root, text='Cancel', name='cancel')
    cancel_btn.pack(pady=(0, 8))

    host = TkHost(root)

    asyncio.start_guest_run(
        count, progress, root,
        run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
        done_callback=host.done_callback,
    )

    root.mainloop()


if __name__ == '__main__':
    main()
