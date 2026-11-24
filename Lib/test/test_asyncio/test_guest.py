"""Tests for asyncio.start_guest_run()."""

import asyncio
import gc
import logging
import queue
import signal
import socket
import sys
import threading
import time
import unittest
from functools import partial
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

    def _check_unstarted_run_unwound(self, run_sync_soon_threadsafe,
                                     exc_type):
        # The run fails to start: the exception propagates, and the run
        # leaves nothing behind.  Neither the main task's coroutine nor
        # one of a task created with it has run, both are closed, no task
        # is left pending, and the loop is closed.
        started = []

        async def side():
            started.append('side')

        async def main():
            started.append('main')

        coros = []

        def async_fn():
            coros.append(side())
            asyncio.get_running_loop().create_task(coros[-1])
            coros.append(main())
            return coros[-1]

        loops = []
        new_event_loop = asyncio.events.new_event_loop

        def capture_loop():
            loops.append(new_event_loop())
            return loops[-1]

        with self.assertNoLogs('asyncio', level='ERROR'):
            with mock.patch.object(asyncio.events, 'new_event_loop',
                                   capture_loop):
                with self.assertRaises(exc_type):
                    asyncio.start_guest_run(
                        async_fn,
                        run_sync_soon_threadsafe=run_sync_soon_threadsafe,
                        done_callback=lambda task: self.fail('done_callback'),
                    )
            gc.collect()
        self.assertEqual(started, [])
        for coro in coros:
            self.assertIsNone(coro.cr_frame)  # closed
        [loop] = loops
        self.assertTrue(loop.is_closed())


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

    def test_host_function_raises_at_start(self):
        class HostGone(Exception):
            pass

        def run_sync_soon_threadsafe(fn):
            raise HostGone

        self._check_unstarted_run_unwound(run_sync_soon_threadsafe, HostGone)

    def test_unwind_under_asyncio_host(self):
        # The host may be an asyncio loop, with start_guest_run() called
        # from one of its tasks: the unwind hides the host's loop and its
        # current task, so the guest's tasks can be cancelled (without
        # the latter, the unwind waits forever), and the host function's
        # exception propagates.
        class HostGone(Exception):
            pass

        def run_sync_soon_threadsafe(fn):
            raise HostGone

        async def main():
            host_loop = asyncio.get_running_loop()
            self._check_unstarted_run_unwound(
                run_sync_soon_threadsafe, HostGone)
            self.assertIs(asyncio.get_running_loop(), host_loop)

        asyncio.run(main())

    def test_batches_in_host_task(self):
        # An asyncio host may run posted callables from one of its tasks,
        # so that guest batches run while a host task is current: the
        # guest's tasks must still step, with themselves as current task.
        async def guest():
            await asyncio.sleep(0.01)
            return asyncio.current_task()

        async def pump(queue):
            while True:
                (await queue.get())()

        async def main():
            nonlocal main_task
            main_task = asyncio.current_task()
            host_loop = asyncio.get_running_loop()
            queue = asyncio.Queue()
            done = host_loop.create_future()
            pumping = asyncio.create_task(pump(queue))
            task = asyncio.start_guest_run(
                guest,
                run_sync_soon_threadsafe=lambda fn: (
                    host_loop.call_soon_threadsafe(queue.put_nowait, fn)),
                done_callback=done.set_result,
            )
            try:
                await asyncio.wait_for(done, support.SHORT_TIMEOUT)
            finally:
                pumping.cancel()
            self.assertIs(asyncio.current_task(), main_task)
            return task

        main_task = None
        task = asyncio.run(main())
        self.assertIs(task.result(), task)

    def test_pure_python_current_task(self):
        # With the pure-Python current task functions, which keep the
        # current task per loop, starting from a host callback, where the
        # host loop has no current task, must not try to clear one.
        async def guest():
            await asyncio.sleep(0)
            return 'done'

        async def main():
            host_loop = asyncio.get_running_loop()
            done = host_loop.create_future()

            def start():
                with mock.patch.object(asyncio.tasks, 'current_task',
                                       asyncio.tasks._py_current_task), \
                     mock.patch.object(asyncio.tasks, '_swap_current_task',
                                       asyncio.tasks._py_swap_current_task):
                    try:
                        asyncio.start_guest_run(
                            guest,
                            run_sync_soon_threadsafe=(
                                host_loop.call_soon_threadsafe),
                            done_callback=done.set_result,
                        )
                    except BaseException as exc:
                        done.set_exception(exc)

            host_loop.call_soon(start)
            return await asyncio.wait_for(done, support.SHORT_TIMEOUT)

        task = asyncio.run(main())
        self.assertEqual(task.result(), 'done')

    def test_host_function_runs_callable_at_start(self):
        # A host function that breaks its contract by running the first
        # batch before it returns, and then raises: guest code has run,
        # and may be stuck halfway, so the unwind only closes the loop,
        # rather than run guest code again or wait for it.
        class HostGone(Exception):
            pass

        steps = []

        def run_sync_soon_threadsafe(fn):
            if not steps:
                fn()
                raise HostGone

        async def main():
            steps.append('started')
            try:
                await asyncio.sleep(0)
            finally:
                steps.append('resumed')

        loops = []
        new_event_loop = asyncio.events.new_event_loop

        def capture_loop():
            loops.append(new_event_loop())
            return loops[-1]

        with mock.patch.object(asyncio.events, 'new_event_loop',
                               capture_loop):
            with self.assertRaises(HostGone):
                asyncio.start_guest_run(
                    main,
                    run_sync_soon_threadsafe=run_sync_soon_threadsafe,
                    done_callback=lambda task: self.fail('done_callback'),
                )
        self.assertEqual(steps, ['started'])
        [loop] = loops
        self.assertTrue(loop.is_closed())
        self.assertIsNone(asyncio._get_running_loop())
        with self.assertLogs('asyncio', level='ERROR'):
            # The abandoned main task.
            support.gc_collect()

    def test_posted_finish_after_failed_start(self):
        # A host function that runs the first batch before it returns, in
        # which the main task finishes, queues the run's _finish() and
        # then raises: the run has failed to start, so running the queued
        # callable later does nothing, and done_callback is never called.
        class HostGone(Exception):
            pass

        calls = 0
        posted = []

        def run_sync_soon_threadsafe(fn):
            # Run the first batch, and the one it posts, in which the
            # main task's done callback posts _finish(); queue the rest.
            nonlocal calls
            calls += 1
            if calls > 2:
                posted.append(fn)
                return
            first = calls == 1
            fn()
            if first:
                raise HostGone

        async def main():
            return 'done'

        with self.assertRaises(HostGone):
            asyncio.start_guest_run(
                main,
                run_sync_soon_threadsafe=run_sync_soon_threadsafe,
                done_callback=lambda task: self.fail('done_callback'),
            )
        self.assertTrue(posted)
        for fn in posted:
            fn()

    def test_refused_at_interpreter_shutdown(self):
        with mock.patch.object(asyncio.guest, '_ensure_atexit'), \
             mock.patch.object(asyncio.guest, '_shutting_down', True):
            self._check_unstarted_run_unwound(
                lambda fn: self.fail('posted at shutdown'), RuntimeError)

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
        # Not in debug mode at first, whatever -X dev says: the batches
        # after set_debug(True) see its effect only if the guest's state
        # is kept from batch to batch.
        with debug_mode(False):
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


def debug_mode(enabled):
    """Create event loops in asyncio debug mode, or not, whatever -X dev
    says, as loop.get_debug() reports it."""
    return mock.patch.object(asyncio.coroutines, '_is_debug_mode',
                             lambda: enabled)


async def _call(fn):
    fn()


def _create_task(loop, fn):
    coro = _call(fn)
    try:
        return loop.create_task(coro)
    except BaseException:
        coro.close()
        raise


# The guest loop's non-thread-safe methods that schedule work: called from
# host code between batches, nothing wakes the I/O thread for that work.
HOST_CALLS = {
    'call_soon': lambda loop, fn: loop.call_soon(fn),
    'call_later': lambda loop, fn: loop.call_later(0, fn),
    'call_at': lambda loop, fn: loop.call_at(loop.time(), fn),
    'create_task': _create_task,
}


class TestDebugHostCheck(GuestTestCase):
    """In debug mode, host code that calls the guest loop's non-thread-safe
    methods between batches gets RuntimeError."""

    def _run_with_host_code(self, host_code, *, debug):
        # Runs host_code(loop, task) once, between batches, while the main
        # task waits for nothing else.  host_code must cancel the task.
        host = MockHost()
        ran = []

        def run_host_code():
            self.assertIsNone(asyncio._get_running_loop())
            host_code(task.get_loop(), task)
            ran.append(True)

        async def coro():
            self.assertEqual(asyncio.get_running_loop().get_debug(), debug)
            host.run_sync_soon_threadsafe(run_host_code)
            await asyncio.sleep(3600)

        with debug_mode(debug):
            task = asyncio.start_guest_run(
                coro,
                run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
                done_callback=host.done_callback,
            )
        self.assertIs(host.run(timeout=5), task)
        self.assertTrue(task.cancelled())
        self.assertEqual(ran, [True])

    def test_host_calls_raise(self):
        for name, call in HOST_CALLS.items():
            with self.subTest(name):
                ran = []

                def host_code(loop, task):
                    with self.assertRaisesRegex(
                            RuntimeError,
                            r'from host code.*call_soon_threadsafe\(\)'):
                        call(loop, lambda: ran.append(True))
                    loop.call_soon_threadsafe(task.cancel)

                if name == 'create_task':
                    # The task is made before its first step's call_soon()
                    # raises, and is left pending.
                    logs = self.assertLogs('asyncio', level='ERROR')
                else:
                    logs = self.assertNoLogs('asyncio', level='ERROR')
                with logs as cm:
                    self._run_with_host_code(host_code, debug=True)
                    gc.collect()
                if name == 'create_task':
                    self.assertEqual(
                        [record.getMessage().splitlines()[0]
                         for record in cm.records],
                        ['Task was destroyed but it is pending!'])
                self.assertEqual(ran, [])

    def test_eager_main_task(self):
        # With an eager task factory, the main task may finish inside
        # start_guest_run(): the run adds its done callback as guest code,
        # so the check doesn't fire.
        async def finishes():
            return 'done'

        async def suspends():
            await asyncio.sleep(0)
            return 'done'

        for debug in (False, True):
            for main in (finishes, suspends):
                with self.subTest(debug=debug, main=main.__name__):
                    def async_fn():
                        loop = asyncio.get_running_loop()
                        loop.set_task_factory(asyncio.eager_task_factory)
                        return main()

                    with debug_mode(debug), \
                         self.assertNoLogs('asyncio', level='ERROR'):
                        task = self._run_guest(async_fn)
                    self.assertEqual(task.result(), 'done')
                    self.assertTrue(task.get_loop().is_closed())

    def test_eager_main_task_failure_at_start(self):
        # The main task finished at start, which queued the run's done
        # callback, and then the host function fails: the unwind runs the
        # queued callback, which must not call the host function again.
        class HostGone(Exception):
            pass

        posted = []

        def run_sync_soon_threadsafe(fn):
            posted.append(fn)
            raise HostGone

        async def side():
            await asyncio.sleep(3600)

        async def main():
            return 'done'

        tasks = []

        def async_fn():
            loop = asyncio.get_running_loop()
            loop.set_task_factory(asyncio.eager_task_factory)
            tasks.append(loop.create_task(side()))
            return main()

        for debug in (False, True):
            with self.subTest(debug=debug):
                posted.clear()
                tasks.clear()
                with debug_mode(debug), \
                     self.assertNoLogs('asyncio', level='ERROR'), \
                     self.assertRaises(HostGone):
                    asyncio.start_guest_run(
                        async_fn,
                        run_sync_soon_threadsafe=run_sync_soon_threadsafe,
                        done_callback=lambda task: self.fail('done_callback'),
                    )
                self.assertEqual(len(posted), 1)
                [side_task] = tasks
                self.assertTrue(side_task.cancelled())
                self.assertTrue(side_task.get_loop().is_closed())

    def test_interrupt_hiding_host_at_start(self):
        # A KeyboardInterrupt right after the host's current task is hidden,
        # at start: the run unwinds, and the host's state and current task
        # are restored.
        hide_host_task = asyncio.guest._hide_host_task
        interrupted = []

        def interrupting_hide_host_task(host):
            hide_host_task(host)
            if not interrupted:
                interrupted.append(asyncio.current_task())
                raise KeyboardInterrupt

        loops = []
        new_event_loop = asyncio.events.new_event_loop

        def capture_loop():
            loops.append(new_event_loop())
            return loops[-1]

        async def guest():
            await asyncio.sleep(3600)

        async def main():
            host_task = asyncio.current_task()
            hooks = sys.get_asyncgen_hooks()
            with debug_mode(True), \
                 mock.patch.object(asyncio.events, 'new_event_loop',
                                   capture_loop), \
                 mock.patch.object(asyncio.guest, '_hide_host_task',
                                   interrupting_hide_host_task), \
                 self.assertRaises(KeyboardInterrupt):
                asyncio.start_guest_run(
                    guest,
                    run_sync_soon_threadsafe=lambda fn: self.fail('posted'),
                    done_callback=lambda task: self.fail('done_callback'),
                )
            self.assertIs(asyncio.current_task(), host_task)
            self.assertEqual(sys.get_asyncgen_hooks(), hooks)
            return asyncio.get_running_loop()

        host_loop = asyncio.run(main())
        # The host task was hidden when the interrupt came.
        self.assertEqual(interrupted, [None])
        [guest_loop] = loops
        self.assertIsNot(guest_loop, host_loop)
        self.assertTrue(guest_loop.is_closed())
        self.assertIsNone(asyncio._get_running_loop())
        asyncio.run(asyncio.sleep(0))

    def _check_start_fails_before_main_task(self, exc_type, *,
                                            setup_then=None, async_fn,
                                            host_loop=None):
        # The run fails before it has a main task: the exception propagates
        # as it is, the loop is closed, and the host's state is intact.
        loops = []
        new_event_loop = asyncio.events.new_event_loop

        def capture_loop():
            loop = new_event_loop()
            loops.append(loop)
            if setup_then is not None:
                setup = loop._run_forever_setup

                def run_forever_setup():
                    setup()
                    setup_then()

                loop._run_forever_setup = run_forever_setup
            return loop

        def firstiter(agen):
            pass

        def finalizer(agen):
            pass

        old_hooks = sys.get_asyncgen_hooks()
        old_depth = sys.get_coroutine_origin_tracking_depth()
        sys.set_asyncgen_hooks(firstiter=firstiter, finalizer=finalizer)
        sys.set_coroutine_origin_tracking_depth(3)
        try:
            with debug_mode(True), \
                 self.assertNoLogs('asyncio', level='ERROR'), \
                 mock.patch.object(asyncio.events, 'new_event_loop',
                                   capture_loop), \
                 self.assertRaises(exc_type) as cm:
                asyncio.start_guest_run(
                    async_fn,
                    run_sync_soon_threadsafe=lambda fn: self.fail('posted'),
                    done_callback=lambda task: self.fail('done_callback'),
                )
            self.assertIs(type(cm.exception), exc_type)
            self.assertIs(asyncio._get_running_loop(), host_loop)
            self.assertEqual(sys.get_asyncgen_hooks(), (firstiter, finalizer))
            self.assertEqual(sys.get_coroutine_origin_tracking_depth(), 3)
        finally:
            sys.set_asyncgen_hooks(*old_hooks)
            sys.set_coroutine_origin_tracking_depth(old_depth)
        [loop] = loops
        self.assertFalse(loop.is_running())
        self.assertTrue(loop.is_closed())

    def test_async_fn_raises(self):
        class Boom(Exception):
            pass

        def async_fn():
            raise Boom

        self._check_start_fails_before_main_task(Boom, async_fn=async_fn)

    def _check_tasks_cancelled_before_main_task(self, exc_type, async_fn,
                                                 started, logged=()):
        # The run fails before it has a main task, after guest code made
        # a task: it is cancelled before its coroutine runs, so it isn't
        # left pending.  Only the messages in logged are logged.
        loops = []
        new_event_loop = asyncio.events.new_event_loop

        def capture_loop():
            loops.append(new_event_loop())
            return loops[-1]

        for debug in (False, True):
            with self.subTest(debug=debug):
                started.clear()
                loops.clear()
                hooks = sys.get_asyncgen_hooks()
                with debug_mode(debug), \
                     mock.patch.object(asyncio.events, 'new_event_loop',
                                       capture_loop), \
                     self.assertLogs('asyncio', level='ERROR') as cm:
                    with self.assertRaises(exc_type):
                        asyncio.start_guest_run(
                            async_fn,
                            run_sync_soon_threadsafe=(
                                lambda fn: self.fail('posted')),
                            done_callback=(
                                lambda task: self.fail('done_callback')),
                        )
                    gc.collect()
                    # assertLogs() needs a record.
                    logging.getLogger('asyncio').error('end')
                self.assertEqual(
                    [record.getMessage().splitlines()[0]
                     for record in cm.records],
                    [*logged, 'end'])
                self.assertEqual(started, [])
                [loop] = loops
                self.assertTrue(loop.is_closed())
                self.assertIsNone(asyncio._get_running_loop())
                self.assertEqual(sys.get_asyncgen_hooks(), hooks)

    def test_async_fn_raises_after_creating_task(self):
        class Boom(Exception):
            pass

        started = []

        async def side():
            started.append('side')

        def async_fn():
            asyncio.get_running_loop().create_task(side())
            raise Boom

        self._check_tasks_cancelled_before_main_task(Boom, async_fn, started)

    def test_eager_main_task_interrupted(self):
        # The main task's first step raises KeyboardInterrupt, which
        # create_task() propagates, so there is no main task, after the
        # step made a task.
        started = []

        async def side():
            started.append('side')

        async def main():
            asyncio.get_running_loop().create_task(side())
            raise KeyboardInterrupt

        def eager_once(loop, coro, **kwargs):
            # Only the main task is eager: side() must not run at once.
            loop.set_task_factory(None)
            return asyncio.Task(coro, loop=loop, eager_start=True, **kwargs)

        def async_fn():
            asyncio.get_running_loop().set_task_factory(eager_once)
            return main()

        # The main task holds the KeyboardInterrupt, which nothing
        # retrieves, as when a task's step raises one under asyncio.run().
        self._check_tasks_cancelled_before_main_task(
            KeyboardInterrupt, async_fn, started,
            logged=['Task exception was never retrieved'])

    def test_async_fn_raises_under_asyncio_host(self):
        class Boom(Exception):
            pass

        def async_fn():
            raise Boom

        async def main():
            host_task = asyncio.current_task()
            self._check_start_fails_before_main_task(
                Boom, async_fn=async_fn,
                host_loop=asyncio.get_running_loop())
            self.assertIs(asyncio.current_task(), host_task)

        # In a thread with a timeout: a broken unwind may hang.
        errors = []

        def run():
            try:
                asyncio.run(main())
            except BaseException as exc:
                errors.append(exc)

        # A daemon, so a hung unwind fails this test without also keeping the
        # process from exiting.
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(30)
        self.assertFalse(thread.is_alive(), 'the unwind hangs')
        if errors:
            raise errors[0]

    def test_interrupt_after_setup(self):
        async def main():
            pass

        def interrupt():
            raise KeyboardInterrupt

        self._check_start_fails_before_main_task(
            KeyboardInterrupt, setup_then=interrupt, async_fn=main)

    def test_interrupt_later_at_start(self):
        # An interrupt anywhere at start unwinds the run: here, when the
        # I/O thread is made.
        with mock.patch.object(threading, 'Thread',
                               side_effect=KeyboardInterrupt):
            self._check_unstarted_run_unwound(
                lambda fn: self.fail('posted'), KeyboardInterrupt)

    def test_guest_run_hosted_by_guest_run(self):
        # A run whose host is another guest loop runs its batches, start
        # and cleanup inside the other loop's batches.  Its guest code may
        # call the other loop's methods, as from that loop's guest code.
        async def outer_main():
            outer = asyncio.get_running_loop()
            reply = outer.create_future()
            reply.add_done_callback(lambda fut: None)
            calls = []

            def async_fn():
                outer.call_soon(calls.append, 'start')
                return inner_main()

            async def inner_main():
                outer.call_soon(calls.append, 'batch')
                await asyncio.sleep(0)
                reply.set_result('inner')

            def done_callback(task):
                outer.call_soon(calls.append, 'done')
                done.set_result(task)

            done = outer.create_future()
            asyncio.start_guest_run(
                async_fn,
                run_sync_soon_threadsafe=outer.call_soon_threadsafe,
                done_callback=done_callback,
            )
            inner_task = await done
            await asyncio.sleep(0)
            return inner_task.result(), await reply, calls

        with debug_mode(True):
            task = self._run_guest(outer_main)
        self.assertEqual(task.result(),
                         (None, 'inner', ['start', 'batch', 'done']))

    def test_hosted_run_during_final_cleanup(self):
        # A run hosted by another guest run may call the other loop while
        # that loop's final cleanup runs: there, a task of the other loop,
        # cancelled by the cleanup, cancels the hosted run and waits for
        # it, and the hosted run's guest code calls the other loop.
        log = []

        async def outer_main():
            outer = asyncio.get_running_loop()
            self.assertTrue(outer.get_debug())
            inner_done = outer.create_future()
            started = outer.create_future()

            async def inner_main():
                outer.call_soon_threadsafe(
                    started.set_result, asyncio.current_task())
                try:
                    await asyncio.sleep(3600)
                finally:
                    try:
                        outer.call_soon(log.append, 'called')
                    except RuntimeError as exc:
                        log.append(exc)

            asyncio.start_guest_run(
                inner_main,
                run_sync_soon_threadsafe=outer.call_soon_threadsafe,
                done_callback=inner_done.set_result,
            )
            inner_task = await started

            async def background():
                try:
                    await asyncio.sleep(3600)
                finally:
                    inner_task.get_loop().call_soon_threadsafe(
                        inner_task.cancel)
                    await inner_done

            outer.create_task(background())
            await asyncio.sleep(0.01)
            return 'done'

        with debug_mode(True):
            task = self._run_guest(outer_main)
        self.assertEqual(task.result(), 'done')
        self.assertEqual(log, ['called'])

    def test_eager_main_task_starts_hosted_run(self):
        # The main task is eager, so its first step runs inside
        # start_guest_run(), and starts a run hosted by its loop: that
        # run's guest code may call the hosting loop there too.
        log = []

        async def outer_main():
            outer = asyncio.get_running_loop()
            inner_done = outer.create_future()

            async def inner_main():
                return 'inner'

            def inner_fn():
                outer.call_soon(log.append, 'called')
                return inner_main()

            asyncio.start_guest_run(
                inner_fn,
                run_sync_soon_threadsafe=outer.call_soon_threadsafe,
                done_callback=inner_done.set_result,
            )
            inner_task = await inner_done
            await asyncio.sleep(0)
            return inner_task.result()

        def async_fn():
            loop = asyncio.get_running_loop()
            loop.set_task_factory(asyncio.eager_task_factory)
            return outer_main()

        with debug_mode(True):
            task = self._run_guest(async_fn)
        self.assertEqual(task.result(), 'inner')
        self.assertEqual(log, ['called'])

    def test_host_call_right_after_start(self):
        # Host code that runs once start_guest_run() has returned, before
        # the first batch, is between batches too.
        host = MockHost()

        async def coro():
            await asyncio.sleep(3600)

        with debug_mode(True):
            task = asyncio.start_guest_run(
                coro,
                run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
                done_callback=host.done_callback,
            )
        loop = task.get_loop()
        with self.assertRaisesRegex(RuntimeError, 'from host code'):
            loop.call_soon(lambda: None)
        loop.call_soon_threadsafe(task.cancel)
        host.run(timeout=5)
        self.assertTrue(task.cancelled())

    def test_interrupt_adding_done_callback(self):
        # An interrupt once the main task exists, at start, unwinds the run.
        class InterruptingTask(asyncio.Task):
            interrupted = False

            def add_done_callback(self, *args, **kwargs):
                if not InterruptingTask.interrupted:
                    InterruptingTask.interrupted = True
                    raise KeyboardInterrupt
                return super().add_done_callback(*args, **kwargs)

        coros = []

        async def main():
            pass

        def async_fn():
            loop = asyncio.get_running_loop()
            loop.set_task_factory(
                lambda loop, coro, **kwargs:
                    InterruptingTask(coro, loop=loop, **kwargs))
            coros.append(main())
            return coros[-1]

        loops = []
        new_event_loop = asyncio.events.new_event_loop

        def capture_loop():
            loops.append(new_event_loop())
            return loops[-1]

        with mock.patch.object(asyncio.events, 'new_event_loop',
                                   capture_loop), \
             self.assertRaises(KeyboardInterrupt):
            asyncio.start_guest_run(
                async_fn,
                run_sync_soon_threadsafe=lambda fn: self.fail('posted'),
                done_callback=lambda task: self.fail('done_callback'),
            )
        self.assertTrue(InterruptingTask.interrupted)
        [coro] = coros
        self.assertIsNone(coro.cr_frame)  # closed
        [loop] = loops
        self.assertTrue(loop.is_closed())
        self.assertIsNone(asyncio._get_running_loop())

    @unittest.skipUnless(hasattr(signal, 'set_wakeup_fd'),
                         'requires signal.set_wakeup_fd')
    def test_wakeup_fd_restored_when_loop_creation_fails(self):
        if threading.current_thread() is not threading.main_thread():
            self.skipTest('requires the main thread')
        rsock, wsock = socket.socketpair()
        self.addCleanup(rsock.close)
        self.addCleanup(wsock.close)
        wsock.setblocking(False)
        old_fd = signal.set_wakeup_fd(wsock.fileno())
        self.addCleanup(signal.set_wakeup_fd, old_fd)

        def failing_new_event_loop():
            # As a loop could, before it fails.
            signal.set_wakeup_fd(-1)
            raise KeyboardInterrupt

        with mock.patch.object(asyncio.events, 'new_event_loop',
                               failing_new_event_loop), \
             self.assertRaises(KeyboardInterrupt):
            asyncio.start_guest_run(
                asyncio.sleep, 0,
                run_sync_soon_threadsafe=lambda fn: self.fail('posted'),
                done_callback=lambda task: self.fail('done_callback'),
            )
        fd = signal.set_wakeup_fd(-1)
        if fd != -1:
            signal.set_wakeup_fd(fd)
        self.assertEqual(fd, wsock.fileno())

    @unittest.skipUnless(hasattr(signal, 'set_wakeup_fd'),
                         'requires signal.set_wakeup_fd')
    def test_wakeup_fd_restored_when_restore_interrupted(self):
        # An interrupt in the restore of the host's wakeup fd, once the
        # loop exists, which may have installed its own: the unwind
        # restores the host's, and the interrupt propagates.
        if threading.current_thread() is not threading.main_thread():
            self.skipTest('requires the main thread')
        rsock, wsock = socket.socketpair()
        self.addCleanup(rsock.close)
        self.addCleanup(wsock.close)
        wsock.setblocking(False)
        old_fd = signal.set_wakeup_fd(wsock.fileno())
        self.addCleanup(signal.set_wakeup_fd, old_fd)

        restore_wakeup_fd = asyncio.guest._restore_wakeup_fd
        interrupted = []

        def interrupted_restore_wakeup_fd(fd):
            if not interrupted:
                interrupted.append(True)
                raise KeyboardInterrupt
            restore_wakeup_fd(fd)

        with mock.patch.object(asyncio.guest, '_restore_wakeup_fd',
                               interrupted_restore_wakeup_fd), \
             self.assertRaises(KeyboardInterrupt):
            asyncio.start_guest_run(
                asyncio.sleep, 0,
                run_sync_soon_threadsafe=lambda fn: self.fail('posted'),
                done_callback=lambda task: self.fail('done_callback'),
            )
        self.assertEqual(interrupted, [True])
        fd = signal.set_wakeup_fd(-1)
        if fd != -1:
            signal.set_wakeup_fd(fd)
        self.assertEqual(fd, wsock.fileno())

    def test_async_fn_changes_guest_state(self):
        # What async_fn(*args) changes in the guest's thread state is kept
        # for the batches, and not seen by the host.
        host = MockHost()

        async def coro():
            return sys.get_coroutine_origin_tracking_depth()

        def async_fn():
            sys.set_coroutine_origin_tracking_depth(7)
            return coro()

        old_depth = sys.get_coroutine_origin_tracking_depth()
        asyncio.start_guest_run(
            async_fn,
            run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
            done_callback=host.done_callback,
        )
        self.assertEqual(sys.get_coroutine_origin_tracking_depth(), old_depth)
        task = host.run()
        self.assertEqual(task.result(), 7)
        self.assertEqual(sys.get_coroutine_origin_tracking_depth(), old_depth)

    def test_check_under_asyncio_host(self):
        # With an asyncio host, host code runs with the host's loop as the
        # running loop, rather than none: it is still host code.
        errors = []

        async def main():
            host_loop = asyncio.get_running_loop()
            done = host_loop.create_future()

            def host_code():
                self.assertIs(asyncio.get_running_loop(), host_loop)
                loop = task.get_loop()
                try:
                    loop.call_soon(lambda: None)
                except RuntimeError as exc:
                    errors.append(exc)
                loop.call_soon_threadsafe(task.cancel)

            async def guest():
                host_loop.call_soon_threadsafe(host_code)
                await asyncio.sleep(3600)

            with debug_mode(True):
                task = asyncio.start_guest_run(
                    guest,
                    run_sync_soon_threadsafe=host_loop.call_soon_threadsafe,
                    done_callback=done.set_result,
                )
            await asyncio.wait_for(done, 5)
            return task

        self.assertTrue(asyncio.run(main()).cancelled())
        self.assertEqual(len(errors), 1)
        [error] = errors
        self.assertIn('from host code', str(error))

    def test_guest_code_calls_asyncio_host_loop(self):
        # Guest code may call the non-thread-safe methods of an asyncio
        # host's loop in debug mode: only a guest loop checks for host
        # code, and the host's loop isn't one.
        async def main():
            host_loop = asyncio.get_running_loop()
            self.assertTrue(host_loop.get_debug())
            done = host_loop.create_future()
            called = host_loop.create_future()

            async def guest():
                host_loop.call_soon(called.set_result, 'called')

            asyncio.start_guest_run(
                guest,
                run_sync_soon_threadsafe=host_loop.call_soon_threadsafe,
                done_callback=done.set_result,
            )
            task = await asyncio.wait_for(done, 5)
            return task.result(), await called

        with debug_mode(True):
            self.assertEqual(asyncio.run(main()), (None, 'called'))

    def test_set_debug_during_run(self):
        # Turned on by guest code, debug mode applies to host code in the
        # batches that follow.
        host = MockHost()
        errors = []

        def host_code():
            loop = task.get_loop()
            try:
                loop.call_soon(lambda: None)
            except RuntimeError as exc:
                errors.append(exc)
            loop.call_soon_threadsafe(task.cancel)

        async def coro():
            asyncio.get_running_loop().set_debug(True)
            host.run_sync_soon_threadsafe(host_code)
            await asyncio.sleep(3600)

        with debug_mode(False):
            task = asyncio.start_guest_run(
                coro,
                run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
                done_callback=host.done_callback,
            )
        host.run(timeout=5)
        self.assertTrue(task.cancelled())
        [error] = errors
        self.assertIn('from host code', str(error))

    def test_asyncio_create_task_from_host(self):
        # Unchanged: there is no running loop between batches.
        def host_code(loop, task):
            coro = _call(lambda: None)
            try:
                with self.assertRaisesRegex(RuntimeError, 'no running'):
                    asyncio.create_task(coro)
            finally:
                coro.close()
            loop.call_soon_threadsafe(task.cancel)

        for debug in (False, True):
            with self.subTest(debug=debug):
                self._run_with_host_code(host_code, debug=debug)

    @unittest.skipIf(asyncio.Task is asyncio.tasks._PyTask,
                     'requires the C Task')
    def test_eager_create_task_from_host(self):
        # With an eager task factory, the C Task's first step needs its
        # loop to be the running loop, so asyncio raises in any mode.
        for debug in (False, True):
            with self.subTest(debug=debug):
                ran = []

                def host_code(loop, task):
                    loop.set_task_factory(asyncio.eager_task_factory)
                    with self.assertRaisesRegex(RuntimeError,
                                                'is not the running loop'):
                        _create_task(loop, lambda: ran.append(True))
                    loop.call_soon_threadsafe(task.cancel)

                with self.assertLogs('asyncio', level='ERROR') as cm:
                    self._run_with_host_code(host_code, debug=debug)
                    gc.collect()
                self.assertEqual(
                    [record.getMessage().splitlines()[0]
                     for record in cm.records],
                    ['Task was destroyed but it is pending!'])
                self.assertEqual(ran, [])

    def test_call_soon_threadsafe_from_host(self):
        # Works, and wakes the I/O thread's wait right away: nothing else
        # would, as the main task waits for nothing else.
        host = MockHost()
        started = []

        async def set_result(fut):
            fut.set_result('host')

        def host_code():
            loop = task.get_loop()
            started.append(time.monotonic())
            loop.call_soon_threadsafe(loop.create_task, set_result(fut))

        async def coro():
            nonlocal fut
            fut = asyncio.get_running_loop().create_future()
            host.run_sync_soon_threadsafe(host_code)
            result = await fut
            return result, time.monotonic() - started[0]

        fut = None
        with debug_mode(True):
            task = asyncio.start_guest_run(
                coro,
                run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
                done_callback=host.done_callback,
            )
        host.run(timeout=30)
        result, elapsed = task.result()
        self.assertEqual(result, 'host')
        # Without the wake-up the task would wait forever: the bound only
        # leaves room for a loaded machine.
        self.assertLess(elapsed, 10)

    def test_host_calls_without_debug(self):
        # Outside debug mode nothing changes: the calls are accepted, and
        # their work runs when something wakes the loop.
        for name, call in HOST_CALLS.items():
            with self.subTest(name):
                ran = []

                def host_code(loop, task):
                    call(loop, lambda: ran.append(True))
                    loop.call_soon_threadsafe(cancel_when_ran, loop, task)

                def cancel_when_ran(loop, task):
                    if ran:
                        task.cancel()
                    else:
                        loop.call_soon(cancel_when_ran, loop, task)

                self._run_with_host_code(host_code, debug=False)
                self.assertEqual(ran, [True])

    def test_guest_calls_unaffected(self):
        async def coro():
            loop = asyncio.get_running_loop()
            self.assertTrue(loop.get_debug())
            futs = []
            for call in HOST_CALLS.values():
                fut = loop.create_future()
                futs.append(fut)
                call(loop, partial(fut.set_result, None))
            await asyncio.gather(*futs)
            await asyncio.create_task(asyncio.sleep(0.01))
            return len(futs)

        with debug_mode(True):
            task = self._run_guest(coro)
        self.assertEqual(task.result(), len(HOST_CALLS))

    def test_other_thread_still_checked(self):
        # Calls from another thread fail as on any loop in debug mode.
        errors = []

        async def coro():
            loop = asyncio.get_running_loop()
            thread = threading.Thread(target=call_from_thread, args=[loop])
            thread.start()
            await asyncio.sleep(0.01)
            thread.join()

        def call_from_thread(loop):
            try:
                loop.call_soon(lambda: None)
            except RuntimeError as exc:
                errors.append(exc)

        with debug_mode(True):
            self.assertIsNone(self._run_guest(coro).result())
        [error] = errors
        self.assertIn('other than the current one', str(error))

    # -- the run's own calls, in debug mode ----------------------------

    def test_completion(self):
        async def background():
            await asyncio.sleep(3600)

        async def agen():
            try:
                yield 1
            finally:
                await asyncio.sleep(0)

        async def coro():
            asyncio.create_task(background())
            it = agen()
            holder.append(it)
            await anext(it)
            await asyncio.sleep(0.01)
            return 'done'

        holder = []
        with debug_mode(True), self.assertNoLogs('asyncio', level='ERROR'):
            task = self._run_guest(coro)
        self.assertEqual(task.result(), 'done')

    def test_exception(self):
        async def coro():
            await asyncio.sleep(0)
            raise ValueError('boom')

        with debug_mode(True):
            task = self._run_guest(coro)
        with self.assertRaisesRegex(ValueError, 'boom'):
            task.result()

    def test_cancel_from_host(self):
        def host_code(loop, task):
            loop.call_soon_threadsafe(task.cancel)

        self._run_with_host_code(host_code, debug=True)

    def test_failure_at_start(self):
        class HostGone(Exception):
            pass

        def run_sync_soon_threadsafe(fn):
            raise HostGone

        with debug_mode(True):
            self._check_unstarted_run_unwound(
                run_sync_soon_threadsafe, HostGone)

    def test_refused_at_interpreter_shutdown(self):
        with debug_mode(True), \
             mock.patch.object(asyncio.guest, '_ensure_atexit'), \
             mock.patch.object(asyncio.guest, '_shutting_down', True):
            self._check_unstarted_run_unwound(
                lambda fn: self.fail('posted at shutdown'),
                RuntimeError)


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
