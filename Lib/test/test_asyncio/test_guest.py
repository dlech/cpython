"""Tests for asyncio.start_guest_run()."""

import asyncio
import queue
import signal
import socket
import sys
import threading
import time
import unittest
from unittest import mock
from test import support
from test.support import threading_helper
from test.support.script_helper import assert_python_ok

threading_helper.requires_working_threading(module=True)


def tearDownModule():
    asyncio.set_event_loop(None)


class MockHost:
    """A minimal host event loop that uses a thread-safe queue.

    Simulates a GUI toolkit main loop without any actual GUI dependency.
    Callbacks are collected in a queue and drained by :meth:`run`.
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._done = threading.Event()
        self._task = None

    def run_sync_soon_threadsafe(self, fn):
        self._queue.put(fn)

    def done_callback(self, task):
        self._task = task
        self._done.set()

    def run(self, timeout=10.0):
        """Drain callbacks until *done_callback* fires or *timeout* expires."""
        deadline = time.monotonic() + timeout
        while not self._done.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("MockHost.run() timed out")
            try:
                fn = self._queue.get(timeout=min(remaining, 0.05))
                fn()
            except queue.Empty:
                pass
        # Drain any trailing callbacks.
        while True:
            try:
                fn = self._queue.get_nowait()
                fn()
            except queue.Empty:
                break
        return self._task


class GuestTestCase(unittest.TestCase):

    def setUp(self):
        self._thread_key = threading_helper.threading_setup()

    def tearDown(self):
        threading_helper.threading_cleanup(*self._thread_key)

    def _run_guest(self, async_fn, *args, timeout=10.0):
        """Helper: run *async_fn* in guest mode and return the completed task."""
        host = MockHost()
        asyncio.start_guest_run(
            async_fn, *args,
            run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
            done_callback=host.done_callback,
        )
        return host.run(timeout=timeout)


class TestGuestRun(GuestTestCase):
    """Test asyncio.start_guest_run with a mock host loop."""

    # -- basic lifecycle -----------------------------------------------

    def test_simple_return(self):
        async def coro():
            return 42

        task = self._run_guest(coro)
        self.assertTrue(task.done())
        self.assertEqual(task.result(), 42)

    def test_return_none(self):
        async def coro():
            pass

        task = self._run_guest(coro)
        self.assertIsNone(task.result())

    def test_with_args(self):
        async def add(a, b):
            return a + b

        task = self._run_guest(add, 3, 7)
        self.assertEqual(task.result(), 10)

    def test_early_sync_completion(self):
        # The task can finish in the very first batch.
        async def coro():
            return 'early'

        host = MockHost()
        task = asyncio.start_guest_run(
            coro,
            run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
            done_callback=host.done_callback,
        )
        self.assertIs(host.run(), task)
        self.assertEqual(task.result(), 'early')

    # -- exception propagation -----------------------------------------

    def test_exception(self):
        async def coro():
            raise ValueError("boom")

        task = self._run_guest(coro)
        self.assertTrue(task.done())
        with self.assertRaises(ValueError) as cm:
            task.result()
        self.assertEqual(str(cm.exception), "boom")

    # -- cancellation --------------------------------------------------

    def test_cancel_from_host(self):
        host = MockHost()

        def cancel_from_host():
            # Host code, e.g. a button handler.  Use call_soon_threadsafe
            # to wake the I/O thread's selector.
            task.get_loop().call_soon_threadsafe(task.cancel)

        async def coro():
            host.run_sync_soon_threadsafe(cancel_from_host)
            await asyncio.sleep(3600)

        task = asyncio.start_guest_run(
            coro,
            run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
            done_callback=host.done_callback,
        )
        host.run(timeout=5)
        self.assertTrue(task.cancelled())

    def test_starts_after_return(self):
        started = False

        async def coro():
            nonlocal started
            started = True

        host = MockHost()
        asyncio.start_guest_run(
            coro,
            run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
            done_callback=host.done_callback,
        )
        self.assertFalse(started)
        host.run()
        self.assertTrue(started)

    # -- asyncio primitives work inside guest --------------------------

    def test_sleep(self):
        async def coro():
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await asyncio.sleep(0.1)
            return loop.time() - t0

        task = self._run_guest(coro)
        self.assertGreaterEqual(task.result(), 0.05)

    def test_create_task(self):
        async def helper():
            await asyncio.sleep(0.01)
            return "helper"

        async def coro():
            t = asyncio.ensure_future(helper())
            return await t

        task = self._run_guest(coro)
        self.assertEqual(task.result(), "helper")

    def test_gather(self):
        async def sleeper(n):
            await asyncio.sleep(0.01 * n)
            return n

        async def coro():
            return await asyncio.gather(sleeper(1), sleeper(2), sleeper(3))

        task = self._run_guest(coro)
        self.assertEqual(task.result(), [1, 2, 3])

    def test_call_later(self):
        async def coro():
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            loop.call_later(0.05, fut.set_result, "later")
            return await fut

        task = self._run_guest(coro)
        self.assertEqual(task.result(), "later")

    def test_call_soon_threadsafe(self):
        timer = None

        async def coro():
            nonlocal timer
            loop = asyncio.get_running_loop()
            fut = loop.create_future()

            def setter():
                loop.call_soon_threadsafe(fut.set_result, "safe")
            timer = threading.Timer(0.05, setter)
            timer.start()
            return await fut

        task = self._run_guest(coro)
        self.assertEqual(task.result(), "safe")
        timer.join()

    def test_socket_io(self):
        async def coro():
            a, b = socket.socketpair()
            ra, wa = await asyncio.open_connection(sock=a)
            rb, wb = await asyncio.open_connection(sock=b)
            try:
                for i in range(20):
                    wa.write(b'%d\n' % i)
                    line = await rb.readline()
                    wb.write(line)
                    self.assertEqual(await ra.readline(), b'%d\n' % i)
            finally:
                wa.close()
                wb.close()
                await wa.wait_closed()
                await wb.wait_closed()
            return "ok"

        task = self._run_guest(coro)
        self.assertEqual(task.result(), "ok")

    def _threads_calling(self, cls, name):
        # Patches cls.name to record the threads that call it.
        original = getattr(cls, name)
        threads = set()

        def recording(self, *args, **kwargs):
            threads.add(threading.get_ident())
            return original(self, *args, **kwargs)

        return mock.patch.object(cls, name, recording), threads

    @unittest.skipUnless(sys.platform == 'win32', 'IOCP only')
    def test_io_processed_on_host_thread(self):
        # Data arrives while the I/O thread waits for it.  The I/O thread
        # only waits: what arrived is processed on the host thread.
        from asyncio import windows_events
        patch, threads = self._threads_calling(
            windows_events.IocpProactor, '_process_completion_status')

        async def coro():
            loop = asyncio.get_running_loop()
            a, b = socket.socketpair()
            a.setblocking(False)
            try:
                for i in range(3):
                    timer = threading.Timer(0.05, b.send, [b'%d' % i])
                    timer.start()
                    self.assertEqual(await loop.sock_recv(a, 10), b'%d' % i)
                    timer.join()
            finally:
                a.close()
                b.close()
            return "ok"

        with patch:
            task = self._run_guest(coro)
        self.assertEqual(task.result(), "ok")
        self.assertEqual(threads, {threading.get_ident()})

    @unittest.skipUnless(sys.platform == 'win32', 'IOCP only')
    def test_abort_processes_events_of_failed_poll(self):
        # If the I/O thread's wait fails after taking completions off the
        # port, _abort() processes them: their futures complete, and closing
        # the loop doesn't wait for them forever.
        from asyncio import base_events, windows_events
        wait = windows_events.IocpProactor._wait_for_completion_statuses

        def failing_wait(self, timeout=None):
            # Fails once the recv's completion is among the statuses.
            statuses = wait(self, timeout)
            if (any(status[3] == recv_address for status in statuses) and
                    threading.current_thread().name == 'asyncio-guest-io'):
                error = OSError(22, 'injected')
                base_events._attach_events(error, statuses)
                raise error
            return statuses

        errors = []
        recv = recv_address = None

        async def coro():
            nonlocal recv, recv_address
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(
                lambda loop, context: errors.append(context))
            a, b = socket.socketpair()
            try:
                recv = loop._proactor.recv(a, 10)
                recv_address = recv._ov.address
                timer = threading.Timer(0.05, b.send, [b'data'])
                timer.start()
                try:
                    await recv
                finally:
                    timer.join()
            finally:
                a.close()
                b.close()

        close = windows_events.IocpProactor.close

        def checked_close(self):
            # Without the fix the recv's completion is dropped, so close()
            # would wait for it forever.  The kernel is done with it (its
            # status was taken off the port), so forget it: the assertions
            # below then fail rather than hang.
            if recv is not None and recv.cancelled():
                self._cache.pop(recv_address, None)
            close(self)

        with (mock.patch.object(windows_events.IocpProactor,
                                '_wait_for_completion_statuses',
                                failing_wait),
              mock.patch.object(windows_events.IocpProactor, 'close',
                                checked_close)):
            task = self._run_guest(coro)
        self.assertTrue(task.cancelled())
        self.assertEqual(recv.result(), b'data')
        self.assertEqual(
            [context['message'] for context in errors],
            ['asyncio guest I/O thread failed'])
        # The error's traceback holds frames of the I/O thread and the
        # task: let go of them, so the threads don't outlive the test.
        errors.clear()
        task = recv = None
        support.gc_collect()

    def test_timers_read_on_host_thread(self):
        # The I/O thread only waits: the host computes how long, so only
        # the host thread touches the loop's timers.
        from asyncio import base_events
        patch, threads = self._threads_calling(
            base_events.BaseEventLoop, '_poll_timeout')

        async def coro():
            for _ in range(3):
                await asyncio.sleep(0.02)
            return "ok"

        with patch:
            task = self._run_guest(coro)
        self.assertEqual(task.result(), "ok")
        self.assertEqual(threads, {threading.get_ident()})

    # -- polling -------------------------------------------------------

    def test_ready_callbacks_do_not_go_through_io_thread(self):
        # While callbacks are ready the poll timeout is zero; such polls
        # run on the host thread instead of being handed to the I/O thread.
        if sys.platform == 'win32':
            loop_class = asyncio.windows_events.ProactorEventLoop
        else:
            loop_class = asyncio.BaseEventLoop
        orig_poll_events = loop_class._poll_events
        polls = io_polls = 0

        def poll_events(loop, timeout):
            nonlocal polls, io_polls
            polls += 1
            if threading.current_thread().name == 'asyncio-guest-io':
                io_polls += 1
            return orig_poll_events(loop, timeout)

        async def coro():
            for _ in range(1000):
                await asyncio.sleep(0)

        with mock.patch.object(loop_class, '_poll_events', poll_events):
            task = self._run_guest(coro)
        self.assertIsNone(task.result())
        self.assertGreater(polls, 0)  # the patch is on the polls used
        self.assertLess(io_polls, 10)

    def test_busy_guest_does_not_starve_host(self):
        # Host-thread polls reschedule themselves through the host loop,
        # so host callbacks still run while guest tasks are busy.
        host = MockHost()
        host_ran = threading.Event()

        async def coro():
            for i in range(100_000):
                if host_ran.is_set():
                    return i
                await asyncio.sleep(0)
            return None

        asyncio.start_guest_run(
            coro,
            run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
            done_callback=host.done_callback,
        )
        host.run_sync_soon_threadsafe(host_ran.set)
        task = host.run()
        self.assertIsNotNone(task.result())

    # -- thread lifecycle ----------------------------------------------

    def test_io_thread_nondaemon_and_joined(self):
        seen = {}

        async def coro():
            # The I/O thread is started after the initial batch; a sleep
            # guarantees it is up and polling by the time we look.
            await asyncio.sleep(0.01)
            for thread in threading.enumerate():
                if thread.name == 'asyncio-guest-io':
                    seen['thread'] = thread

        task = self._run_guest(coro)
        self.assertIsNone(task.exception())
        self.assertIn('thread', seen)
        self.assertFalse(seen['thread'].daemon)
        self.assertFalse(seen['thread'].is_alive())

    def test_interpreter_exit_with_pending_run(self):
        # Exiting with an unfinished guest run must not hang: the atexit
        # hook wakes the non-daemon I/O thread out of its selector wait
        # and joins it.
        code = (
            'import asyncio, collections\n'
            'q = collections.deque()\n'
            'async def coro():\n'
            '    await asyncio.sleep(3600)\n'
            'asyncio.start_guest_run(\n'
            '    coro,\n'
            '    run_sync_soon_threadsafe=q.append,\n'
            '    done_callback=lambda task: None,\n'
            ')\n'
        )
        assert_python_ok('-c', code)

    # -- running-loop semantics ----------------------------------------

    def test_is_running_inside(self):
        async def coro():
            return asyncio.get_running_loop().is_running()

        task = self._run_guest(coro)
        self.assertTrue(task.result())

    def test_nested_run_raises(self):
        test = self

        async def coro():
            loop = asyncio.get_running_loop()
            inner = asyncio.sleep(0)
            try:
                with test.assertRaises(RuntimeError):
                    loop.run_until_complete(inner)
            finally:
                inner.close()
            inner = asyncio.sleep(0)
            try:
                with test.assertRaises(RuntimeError):
                    asyncio.run(inner)
            finally:
                inner.close()

        task = self._run_guest(coro)
        self.assertIsNone(task.exception())

    def test_state_restored_after_run(self):
        old_hooks = sys.get_asyncgen_hooks()

        async def coro():
            pass

        task = self._run_guest(coro)
        self.assertEqual(sys.get_asyncgen_hooks(), old_hooks)
        self.assertIsNone(asyncio._get_running_loop())
        self.assertFalse(task.get_loop().is_running())

    # -- per-thread state isolation ------------------------------------

    def test_host_state_between_batches(self):
        # Host code sees the host's running loop and asyncgen hooks, not
        # the guest's, between guest batches.
        def firstiter(agen):
            pass

        def finalizer(agen):
            pass

        host = MockHost()
        seen = {}

        def host_code():
            seen['running_loop'] = asyncio._get_running_loop()
            seen['hooks'] = sys.get_asyncgen_hooks()

        async def coro():
            host.run_sync_soon_threadsafe(host_code)
            await asyncio.sleep(0.01)
            return sys.get_asyncgen_hooks().firstiter

        old_hooks = sys.get_asyncgen_hooks()
        sys.set_asyncgen_hooks(firstiter=firstiter, finalizer=finalizer)
        try:
            task = asyncio.start_guest_run(
                coro,
                run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
                done_callback=host.done_callback,
            )
            host.run()
            self.assertEqual(sys.get_asyncgen_hooks(),
                             (firstiter, finalizer))
        finally:
            sys.set_asyncgen_hooks(*old_hooks)
        self.assertIsNone(seen['running_loop'])
        self.assertEqual(seen['hooks'], (firstiter, finalizer))
        self.assertEqual(task.result(),
                         task.get_loop()._asyncgen_firstiter_hook)

    def test_set_debug_during_run(self):
        # loop.set_debug() changes the guest's coroutine origin tracking
        # for later batches, but not the host's.
        host = MockHost()
        host_depths = []

        def host_code():
            host_depths.append(sys.get_coroutine_origin_tracking_depth())

        async def coro():
            loop = asyncio.get_running_loop()
            loop.set_debug(True)
            await asyncio.sleep(0)
            host.run_sync_soon_threadsafe(host_code)
            await asyncio.sleep(0.01)
            depth = sys.get_coroutine_origin_tracking_depth()
            loop.set_debug(False)
            await asyncio.sleep(0)
            return depth

        old_depth = sys.get_coroutine_origin_tracking_depth()
        asyncio.start_guest_run(
            coro,
            run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
            done_callback=host.done_callback,
        )
        task = host.run()
        self.assertGreater(task.result(), 0)
        self.assertEqual(host_depths, [old_depth])
        self.assertEqual(sys.get_coroutine_origin_tracking_depth(), old_depth)

    def test_host_asyncgen_not_adopted(self):
        # An async generator first iterated by host code belongs to the
        # host: the guest loop must not finalize it at shutdown.
        async def agen():
            yield 1
            yield 2

        host = MockHost()
        host_agen = agen()

        def host_code():
            with self.assertRaises(StopIteration):
                host_agen.asend(None).send(None)

        async def coro():
            host.run_sync_soon_threadsafe(host_code)
            await asyncio.sleep(0.01)
            return host_agen in asyncio.get_running_loop()._asyncgens

        asyncio.start_guest_run(
            coro,
            run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
            done_callback=host.done_callback,
        )
        task = host.run()
        self.assertFalse(task.result())
        self.assertIsNotNone(host_agen.ag_frame)  # not closed by the guest
        host_agen.aclose().close()

    def test_asyncio_host(self):
        # The host may itself be an asyncio event loop.
        async def guest():
            loop = asyncio.get_running_loop()
            await asyncio.sleep(0.01)
            return loop, asyncio.current_task()

        async def main():
            host_loop = asyncio.get_running_loop()
            done = host_loop.create_future()
            task = asyncio.start_guest_run(
                guest,
                run_sync_soon_threadsafe=host_loop.call_soon_threadsafe,
                done_callback=done.set_result,
            )
            self.assertIs(asyncio.get_running_loop(), host_loop)
            await asyncio.sleep(0.01)
            self.assertIs(await done, task)
            self.assertIs(asyncio.get_running_loop(), host_loop)
            return host_loop, task

        host_loop, task = asyncio.run(main())
        guest_loop, current = task.result()
        self.assertIsNot(guest_loop, host_loop)
        self.assertIs(guest_loop, task.get_loop())
        self.assertIs(current, task)
        self.assertTrue(guest_loop.is_closed())

    # -- signal handling -----------------------------------------------

    @unittest.skipUnless(hasattr(signal, 'SIGUSR1'),
                         'requires UNIX signal handling')
    def test_add_signal_handler_raises(self):
        async def coro():
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGUSR1, lambda: None)

        task = self._run_guest(coro)
        with self.assertRaisesRegex(RuntimeError, 'guest mode'):
            task.result()

    @unittest.skipUnless(hasattr(signal, 'SIGUSR1'),
                         'requires UNIX signal handling')
    def test_remove_signal_handler_raises(self):
        async def coro():
            loop = asyncio.get_running_loop()
            loop.remove_signal_handler(signal.SIGUSR1)

        task = self._run_guest(coro)
        with self.assertRaisesRegex(RuntimeError, 'guest mode'):
            task.result()

    @unittest.skipUnless(hasattr(signal, 'set_wakeup_fd'),
                         'requires signal.set_wakeup_fd')
    def test_wakeup_fd_preserved(self):
        if threading.current_thread() is not threading.main_thread():
            self.skipTest('requires the main thread')
        rsock, wsock = socket.socketpair()
        self.addCleanup(rsock.close)
        self.addCleanup(wsock.close)
        wsock.setblocking(False)
        old_fd = signal.set_wakeup_fd(wsock.fileno())
        self.addCleanup(signal.set_wakeup_fd, old_fd)

        async def coro():
            await asyncio.sleep(0.01)

        self._run_guest(coro)

        fd = signal.set_wakeup_fd(-1)
        if fd != -1:
            signal.set_wakeup_fd(fd)
        self.assertEqual(fd, wsock.fileno())

    # -- final cleanup matches asyncio.run() ---------------------------

    def test_background_task_cancelled_on_finish(self):
        state = {}

        async def background():
            await asyncio.sleep(3600)

        async def coro():
            state['bg'] = asyncio.get_running_loop().create_task(background())
            await asyncio.sleep(0.01)

        task = self._run_guest(coro)
        self.assertIsNone(task.exception())
        self.assertTrue(state['bg'].cancelled())

    def test_abandoned_asyncgen_finalized(self):
        finalized = False
        holder = []

        async def agen():
            nonlocal finalized
            try:
                yield 1
            finally:
                finalized = True

        async def coro():
            it = agen()
            holder.append(it)  # keep it alive until shutdown_asyncgens()
            await anext(it)

        task = self._run_guest(coro)
        self.assertIsNone(task.exception())
        self.assertTrue(finalized)

    def test_loop_closed_in_done_callback(self):
        # Cleanup (cancel remaining tasks, close the loop) happens
        # before done_callback, like asyncio.run().
        seen = {}
        host = MockHost()
        original = host.done_callback

        def done_callback(task):
            seen['closed'] = task.get_loop().is_closed()
            original(task)

        async def coro():
            pass

        asyncio.start_guest_run(
            coro,
            run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
            done_callback=done_callback,
        )
        host.run()
        self.assertTrue(seen['closed'])

    def test_loop_closed_after_run(self):
        async def coro():
            pass

        task = self._run_guest(coro)
        self.assertTrue(task.get_loop().is_closed())


class TestBaseEventLoopDecomposition(GuestTestCase):
    """Verify that poll_events / process_events / process_ready exist
    and compose correctly (i.e. _run_once still works)."""

    def test_methods_exist(self):
        loop = asyncio.new_event_loop()
        try:
            self.assertTrue(hasattr(loop, 'poll_events'))
            self.assertTrue(hasattr(loop, 'process_events'))
            self.assertTrue(hasattr(loop, 'process_ready'))
        finally:
            loop.close()

    def test_run_once_still_works(self):
        """asyncio.run() exercises _run_once(); ensure it still functions
        after the refactor."""
        async def coro():
            await asyncio.sleep(0)
            return "ok"

        result = asyncio.run(coro())
        self.assertEqual(result, "ok")

    def test_run_once_uses_overrides(self):
        # A selector loop's _run_once() goes through poll_events() and
        # process_events(), so overriding them affects a running loop too.
        calls = []

        class Loop(asyncio.SelectorEventLoop):
            def process_events(self, event_list):
                calls.append(event_list)
                super().process_events(event_list)

        loop = Loop()
        try:
            loop.run_until_complete(asyncio.sleep(0))
        finally:
            loop.close()
        self.assertTrue(calls)


if __name__ == '__main__':
    unittest.main()
