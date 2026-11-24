"""Support for running asyncio as a guest inside another event loop.

This module provides start_guest_run(), which allows asyncio to run
cooperatively inside a host event loop such as a GUI toolkit's main loop.
The host loop stays in control of its thread while asyncio tasks execute
through a dual-thread architecture:

  Host thread:  process_events() + process_ready() -> hand token to I/O thread
  I/O thread:   wait for token -> poll_events() -> hand events to host

A single "token" (semaphore permit) ping-pongs between the two threads,
so exactly one of them touches the event loop at any moment.  The token
only goes to the I/O thread when the poll would block: if callbacks are
ready or I/O events are pending, the host keeps it and schedules the
next batch on itself.

Inspired by Trio's guest mode (trio.lowlevel.start_guest_run).  The
asyncio-guest project, which this implementation grew out of, has
runnable examples for Tkinter, Qt, GTK, pygame, Win32 and Tornado
hosts:

    https://github.com/congzhangzh/asyncio-guest
"""

__all__ = ('start_guest_run',)

import contextlib
import signal
import sys
import threading
from functools import partial

from . import base_events
from . import constants
from . import events
from . import runners
from . import tasks
from .log import logger

# Stop callbacks of unfinished guest runs, so interpreter shutdown can
# unblock and join their (non-daemon) I/O threads.
_lock = threading.Lock()
_stoppers = set()
_shutting_down = False
_atexit_registered = False


def _python_exit():
    """Wake and join the I/O threads of all unfinished guest runs.

    Registered via threading._register_atexit() so it runs before the
    interpreter joins non-daemon threads (the concurrent.futures
    pattern).  Each run is abandoned: pending tasks are not cancelled
    and done_callback is not called.
    """
    global _shutting_down
    with _lock:
        _shutting_down = True
        stoppers = list(_stoppers)
    for stop in stoppers:
        stop()


def _ensure_atexit():
    global _atexit_registered
    with _lock:
        if _shutting_down:
            raise RuntimeError(
                'cannot start an asyncio guest run at interpreter shutdown')
        if not _atexit_registered:
            threading._register_atexit(_python_exit)
            _atexit_registered = True


def _save_wakeup_fd():
    """Return the current signal wakeup fd, or None if it cannot be read.

    There is no getter for the wakeup fd, so briefly swapping it out is
    the only way to read it.  A signal arriving between the two calls
    loses its wakeup byte (its Python-level handler still runs); the
    window is a few instructions wide.
    """
    if (not hasattr(signal, 'set_wakeup_fd')
            or threading.current_thread() is not threading.main_thread()):
        return None
    fd = signal.set_wakeup_fd(-1)
    if fd != -1:
        signal.set_wakeup_fd(fd)
    return fd


def _restore_wakeup_fd(fd):
    if fd is not None:
        signal.set_wakeup_fd(fd)


def _get_thread_state():
    """Return the per-thread state that a running event loop installs."""
    return (events._get_running_loop(),
            sys.get_asyncgen_hooks(),
            sys.get_coroutine_origin_tracking_depth())


def _set_thread_state(state):
    running_loop, asyncgen_hooks, origin_tracking_depth = state
    events._set_running_loop(running_loop)
    sys.set_asyncgen_hooks(*asyncgen_hooks)
    sys.set_coroutine_origin_tracking_depth(origin_tracking_depth)


def _save_host():
    """Return the host's thread state and current task, for
    _restore_host(), without changing them.

    Callers save the host before changing anything, and restore it in a
    'finally' clause: an exception, such as KeyboardInterrupt, may come
    at any point in between.
    """
    state = _get_thread_state()
    host_loop = state[0]
    if host_loop is None:
        return state, None
    return state, tasks.current_task(host_loop)


def _hide_host_task(host):
    """Clear the host loop's current task, if any.

    With the C Task implementation, which keeps the current task per
    thread, guest tasks cannot step while another task is current, as
    one is when guest work runs from a host task, e.g. when
    start_guest_run() is called from one, or the host runs posted
    callables from one.
    """
    host_state, host_task = host
    if host_task is not None:
        tasks._swap_current_task(host_state[0], None)


def _restore_host(host_state, host_task):
    """Restore the host's thread state and current task."""
    _set_thread_state(host_state)
    if host_task is not None:
        tasks._swap_current_task(host_state[0], host_task)


@contextlib.contextmanager
def _host_loop_hidden():
    """Hide the host's running loop and task, if any, and restore them.

    The guest loop must not start, or be driven to completion at
    shutdown, while another loop is the thread's running loop.  An
    asyncio host is suspended in one of its callbacks at that point, so
    the two loops never actually run nested.
    """
    host = _save_host()
    try:
        _hide_host_task(host)
        events._set_running_loop(None)
        yield
    finally:
        _restore_host(*host)


def start_guest_run(async_fn, *args,
                    run_sync_soon_threadsafe,
                    done_callback):
    """Run async_fn(*args) as a guest inside another event loop.

    Must be called from the host event loop's thread.  The host loop
    (e.g. Tkinter's mainloop) remains in control of that thread; asyncio
    I/O polling runs in a background non-daemon thread that is joined
    when the run finishes.  Returns the Task wrapping *async_fn*; to
    cancel it from the host, use loop.call_soon_threadsafe(task.cancel)
    so that the I/O thread is woken from its selector wait.

    run_sync_soon_threadsafe is a callable that schedules a
    zero-argument callable to run on the host loop's thread.  It must be
    thread-safe and must not raise.  It must return before the callable
    runs, even when called on the host thread, and must never wait for
    the callable to run, which can deadlock.  It is called both on the
    host thread, where it must not wait for the host loop at all, and
    from the I/O thread, where it may wait for the host loop to accept
    the callable, but then the run, and the interpreter's exit, hang if
    the host loop stops while it waits.  It need not preserve ordering.
    If it raises when start_guest_run() first calls it, without having
    run the callable, the run's tasks are cancelled and awaited, the
    loop is closed, and the exception propagates.

    done_callback is called on the host thread after the run has fully
    finished: remaining tasks cancelled, asynchronous generators and the
    default executor shut down, and the loop closed (the same cleanup as
    asyncio.run()).  It receives the completed Task as its sole
    argument.

    The guest loop is the thread's running loop, with its async
    generator hooks installed, only while guest code runs, such as in
    batches of callbacks, at start, and in the final cleanup.  The
    host's are restored in between.  So host code keeps its own asyncio
    state, and the host may itself be an asyncio event loop.
    loop.is_running() is true for the whole run.  The main task starts
    running in the first callback passed to run_sync_soon_threadsafe,
    after start_guest_run() has returned, unless the loop has an eager
    task factory, e.g. set by async_fn(*args), which runs the main
    task's first step at once.  If the start then fails, the cancelled
    tasks' cleanup runs inside start_guest_run(), where it must not wait
    for the host.

    Host code that runs between guest batches, such as an event
    handler, schedules guest work with loop.call_soon_threadsafe(), as
    from any other thread, although it runs on the loop's thread.  The
    loop's other methods, such as call_soon() and create_task(), are for
    guest code: from host code, nothing wakes the loop for the work they
    schedule.  In debug mode, call_soon(), call_later(), call_at() and
    create_task() raise RuntimeError when host code calls them between
    batches.  With an eager task factory, create_task() runs the task's
    first step at once instead: with the C Task implementation (the
    default), that raises RuntimeError in any mode, because the loop
    isn't the running loop, while with the pure-Python Task, the step
    runs as host code, which the check may not catch.
    asyncio.create_task() is unchanged: the guest loop isn't the running
    loop there, so it raises RuntimeError, or with an asyncio host,
    creates the task on the host's loop.

    A modal loop entered by host code keeps the run going, as long as it
    runs the host's queue.  Guest code must not enter one: the batch that
    entered it is still running, so the run makes no progress until it
    returns, and if the modal loop runs the host's queue, it may run
    callables the run already posted re-entrantly.  To show a modal
    dialog from guest code, post it to the host with
    run_sync_soon_threadsafe and await its answer.

    Signal handling stays with the host: the guest loop never touches
    signal.set_wakeup_fd(), and loop.add_signal_handler() raises
    RuntimeError.  The host may forward signals into the loop with
    loop.call_soon_threadsafe().

    If the interpreter exits while the run is unfinished, the run is
    abandoned: the I/O thread is woken and joined, pending tasks are not
    cancelled, and done_callback is not called.
    """
    _ensure_atexit()

    def _close_loop():
        # BaseProactorEventLoop.close() resets the wakeup fd to -1 on
        # the main thread; preserve the host's fd across it.
        fd = _save_wakeup_fd()
        try:
            loop.close()
        finally:
            _restore_wakeup_fd(fd)

    shutdown = threading.Event()
    wakeup = threading.Semaphore(0)
    # Until when the I/O thread may wait (loop.time(), or None for no
    # limit), computed by the host before it hands over the token: only
    # the host touches the loop's timers.
    poll_deadline = None
    finished = False
    cleaned_up = False
    # Whether a batch has run: start_guest_run() only expects that if
    # the host function broke its contract by running the callable.
    batch_ran = False
    backend_thread = None

    # -- helpers (host thread unless noted) ----------------------------

    def _enter_guest(host):
        # Install the guest's per-thread state, with no current task.
        # host is from _save_host(), called before, and _leave_guest(host)
        # must follow in a 'finally' clause, even if this is interrupted.
        _hide_host_task(host)
        _set_thread_state(guest_state)
        loop._guest_code_running = True

    def _leave_guest(host):
        # Keep changes made by guest callbacks, e.g. by loop.set_debug().
        # If entering was interrupted, what is kept may be the host's
        # state, but then the batch raises, and the run is abandoned.
        nonlocal guest_state
        loop._guest_code_running = False
        guest_state = _get_thread_state()
        _restore_host(*host)

    def _cleanup_running_state():
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        host = _save_host()
        try:
            _enter_guest(host)
            loop._run_forever_cleanup()
        finally:
            # Not _leave_guest(): the loop's guest code runs to the end of
            # the final cleanup, so _guest_code_running stays set, and a
            # guest run this loop hosts may still call it.
            _restore_host(*host)

    def _join_backend():
        # By the token invariant the I/O thread is parked in
        # wakeup.acquire() here, never in the selector; the
        # _write_to_self() wake-up is defensive insurance.
        shutdown.set()
        wakeup.release()
        try:
            loop._write_to_self()
        except Exception:
            pass
        if backend_thread is not None and backend_thread.ident is not None:
            backend_thread.join()

    def _deliver(task, *, graceful):
        with _lock:
            _stoppers.discard(_stop_at_exit)
        try:
            if graceful:
                # Same cleanup as asyncio.run().
                with _host_loop_hidden():
                    runners._cancel_all_tasks(loop)
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.run_until_complete(
                        loop.shutdown_default_executor(
                            constants.THREAD_JOIN_TIMEOUT))
        finally:
            _close_loop()
            done_callback(task)

    def _finish(task):
        # Always scheduled via run_sync_soon_threadsafe, never called
        # inline from a loop callback: the cleanup below drives the loop
        # with run_until_complete(), which would re-enter the
        # process_ready() iteration such a callback runs in.
        nonlocal finished
        if finished:
            return
        finished = True
        _join_backend()
        _cleanup_running_state()
        _deliver(task, graceful=True)

    def _abort(exc):
        # The I/O thread died; the loop can no longer be driven through
        # the guest handshake.  Unwind the main task directly.
        nonlocal finished
        if finished:
            return
        finished = True
        _join_backend()
        event_list = base_events._events_from(exc)
        if event_list:
            # The failed poll took these events before failing: process
            # them, or their futures would never complete, and closing the
            # loop would wait for them.
            host = _save_host()
            try:
                _enter_guest(host)
                loop.process_events(event_list)
            except Exception as e:
                loop.call_exception_handler({
                    'message': ('asyncio guest run could not process the '
                                'events of its failed poll'),
                    'exception': e,
                })
            finally:
                _leave_guest(host)
        _cleanup_running_state()
        if not main_task.done():
            main_task.cancel(
                msg=f'asyncio guest I/O thread failed: {exc!r}')
            try:
                with _host_loop_hidden():
                    loop.run_until_complete(
                        tasks.gather(main_task, return_exceptions=True))
            except Exception:
                # The loop itself is broken (e.g. the selector raised);
                # abandon any pending work.
                pass
        loop.call_exception_handler({
            'message': 'asyncio guest I/O thread failed',
            'exception': exc,
            'task': main_task,
        })
        _deliver(main_task, graceful=False)

    def _process_on_host(event_list):
        """Process one batch of asyncio work on the host thread."""
        nonlocal poll_deadline, batch_ran
        if shutdown.is_set() or loop.is_closed():
            return
        batch_ran = True
        host = _save_host()
        try:
            _enter_guest(host)
            loop.process_events(event_list)
            event_list = None
            loop.process_ready()
            timeout = loop._poll_timeout()
            deadline = None if timeout is None else loop.time() + timeout
        except BaseException:
            # Internal loop failure (user callback exceptions are routed
            # to the exception handler by Handle._run).  Unblock the I/O
            # thread so it can exit and be joined.
            shutdown.set()
            wakeup.release()
            raise
        finally:
            _leave_guest(host)
        if shutdown.is_set():
            return
        # Fast path: poll without blocking on the host thread.  If there
        # is work to do right away, run the next batch on the host too;
        # a round trip through the I/O thread costs two thread switches.
        try:
            event_list = loop._poll_events(0)
        except Exception as exc:
            # Report it like an I/O thread failure.  Scheduled rather
            # than called inline, so that _abort() can join the I/O
            # thread even when this is the initial batch.
            shutdown.set()
            run_sync_soon_threadsafe(partial(_abort, exc))
            return
        if timeout == 0 or event_list:
            run_sync_soon_threadsafe(partial(_process_on_host, event_list))
            return
        # Hand the polling token back to the I/O thread.  Exactly one
        # token circulates: while the host runs a batch, the I/O thread
        # is parked in wakeup.acquire(), so it can never poll (or touch
        # the loop at all) concurrently with this function.
        poll_deadline = deadline
        wakeup.release()

    def _on_task_done(task):
        if finished:
            # The run failed to start, or is being cleaned up: e.g. an
            # eager main task finished at start, and the unwind ran the
            # callback it had queued.
            return
        # Runs inside process_ready() while the host holds the token,
        # so the I/O thread is parked in wakeup.acquire(): the released
        # token wakes it, it observes 'shutdown', and exits without
        # re-entering the selector.
        shutdown.set()
        wakeup.release()
        try:
            run_sync_soon_threadsafe(partial(_finish, task))
        except Exception as exc:
            loop.call_exception_handler({
                'message': ('asyncio guest run could not schedule its '
                            'completion callback on the host'),
                'exception': exc,
                'task': task,
            })

    def _backend():
        """I/O thread: wait for the token, poll, hand events to the host."""
        try:
            while True:
                wakeup.acquire()
                if shutdown.is_set():
                    return
                # Only wait: the host computed the deadline, so the loop's
                # timers are only ever touched on the host thread.
                if poll_deadline is None:
                    timeout = None
                else:
                    timeout = max(0, poll_deadline - loop.time())
                event_list = loop._poll_events(timeout)
                if shutdown.is_set():
                    # Interpreter exit woke the selector; the host may
                    # be gone, so do not call into it.
                    return
                run_sync_soon_threadsafe(
                    partial(_process_on_host, event_list))
        except Exception as exc:
            if shutdown.is_set():
                return
            shutdown.set()
            try:
                run_sync_soon_threadsafe(partial(_abort, exc))
            except Exception:
                logger.error(
                    'asyncio guest run abandoned: host is unreachable '
                    'after an I/O thread failure', exc_info=True)

    def _stop_at_exit():
        # Interpreter-exit hook: unblock and join the I/O thread so
        # shutdown does not hang on a non-daemon thread.  The run is
        # abandoned -- no cancellation, no callbacks, no loop.close().
        shutdown.set()
        wakeup.release()
        try:
            loop._write_to_self()
        except Exception:
            pass
        if backend_thread.ident is not None:
            backend_thread.join()

    def _unwind_unstarted():
        # The run failed to start: cancel its tasks, and run the loop
        # until the cancellations have finished them, as asyncio.run()
        # does, so nothing is left pending.  Their coroutines haven't run,
        # unless an eager task factory ran a first step, and then the
        # task's cleanup runs here.  If the host function ran a batch
        # anyway, guest code may be stuck halfway, and waiting for it
        # could hang, so just close the loop.  Mark the run finished, so
        # that a _finish() or _abort() it posted does nothing.
        nonlocal finished
        finished = True
        _cleanup_running_state()
        main_task.remove_done_callback(_on_task_done)
        try:
            if not batch_ran:
                with _host_loop_hidden():
                    runners._cancel_all_tasks(loop)
        finally:
            _close_loop()

    # -- start ---------------------------------------------------------

    # Whatever fails, or is interrupted, from here on, unwinds as far as
    # the run got: closes the loop, or cancels the tasks too.
    loop = main_task = host_wakeup_fd = None
    try:
        # Create the loop without letting it capture the signal wakeup
        # fd: on Windows, BaseProactorEventLoop.__init__ installs its
        # self-pipe as the wakeup fd when on the main thread, which would
        # silently break the host's signal handling.
        host_wakeup_fd = _save_wakeup_fd()
        try:
            loop = events.new_event_loop()
        finally:
            _restore_wakeup_fd(host_wakeup_fd)

        # The host owns signal handling (see the docstring).  Never reset:
        # 'finally' blocks of tasks cancelled during the final cleanup
        # must not be able to install signal handlers either.
        loop._guest_mode = True

        # Mark the loop as running for the whole guest run.  On Windows
        # this also starts the proactor's self-reading loop, so that
        # call_soon_threadsafe() can wake the I/O thread's poll.  The
        # per-thread part of the setup is captured as guest_state and
        # swapped in only around guest callbacks (see _enter_guest()).
        with _host_loop_hidden():
            loop._run_forever_setup()
            guest_state = _get_thread_state()
            # Guest code runs here: async_fn(*args), and with an eager task
            # factory, the main task's first step.  If that finishes the
            # task, adding the done callback schedules it with call_soon().
            loop._guest_code_running = True
            try:
                main_task = loop.create_task(async_fn(*args))
                main_task.add_done_callback(_on_task_done)
            finally:
                loop._guest_code_running = False
            # Again: async_fn(*args) may have changed it.
            guest_state = _get_thread_state()

        backend_thread = threading.Thread(
            target=_backend, name='asyncio-guest-io')

        with _lock:
            if _shutting_down:
                raise RuntimeError(
                    'cannot start an asyncio guest run at interpreter '
                    'shutdown')
            _stoppers.add(_stop_at_exit)

        # The I/O thread waits for the token, which the host holds until
        # the first batch has run.  That batch is scheduled rather than
        # run inline: with an asyncio host the caller is usually a host
        # task, and a guest task cannot start while another task runs.
        backend_thread.start()
        run_sync_soon_threadsafe(partial(_process_on_host, []))
    except BaseException:
        if main_task is not None:
            with _lock:
                _stoppers.discard(_stop_at_exit)
            _join_backend()
            _unwind_unstarted()
        elif loop is not None:
            try:
                # Again, in case the start was interrupted before the
                # host's fd was restored: else _close_loop() would save the
                # loop's own, close() would close it, and _close_loop()
                # would then restore the closed fd.
                _restore_wakeup_fd(host_wakeup_fd)
                with _host_loop_hidden():
                    # Whether or not the setup ran, or ran only partway:
                    # the cleanup only resets what the setup may have set.
                    loop._run_forever_cleanup()
                    # Cancel any task guest code made before it failed,
                    # e.g. async_fn(*args), or an eager main task's first
                    # step, as _unwind_unstarted() does.  As guest code:
                    # no batch has run, and the loop is now the running
                    # loop only while this drives it.
                    loop._guest_code_running = True
                    runners._cancel_all_tasks(loop)
            finally:
                _close_loop()
        raise

    return main_task
