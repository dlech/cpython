.. currentmodule:: asyncio

.. _asyncio-guest:

==========
Guest Mode
==========

**Source code:** :source:`Lib/asyncio/guest.py`

----

Running asyncio as a Guest in Another Event Loop
=================================================

*Guest mode* allows asyncio to run cooperatively inside a *host* event loop
such as a GUI toolkit's main loop (Tkinter, Qt, GTK, etc.).  Instead of
replacing the host loop, asyncio piggybacks on it:

* The **host thread** keeps running its own main loop as usual.
* A **background I/O thread** blocks on the selector (I/O polling).
  When I/O events arrive it hands them back to the host thread via a
  thread-safe callback.  The thread is not a daemon thread; it is joined
  when the guest run finishes.
* The host thread then runs
  :meth:`loop.process_events() <asyncio.loop.process_events>` and
  :meth:`loop.process_ready() <asyncio.loop.process_ready>` to advance
  the asyncio event loop by one step, then signals the I/O thread to
  poll again.  If callbacks are already ready or I/O events are pending,
  the host thread skips the I/O thread and schedules the next step on
  itself, so busy tasks do not pay for a thread switch on every step.

Exactly one of the two threads touches the event loop at any moment, so
neither the host loop nor the asyncio loop starves the other.

Typical use cases:

* Incrementally migrating a Tkinter/Qt/GTK application to ``async/await``
  without replacing the native event loop.
* Embedding asyncio I/O (HTTP clients, websockets, …) inside a GUI app.
* Running asyncio alongside a framework that owns the main thread.
* Running a second, isolated asyncio loop inside an asyncio program, for
  example a server under test whose loop is separate from the test's.

.. rubric:: Example

See :source:`Doc/includes/asyncio_guest_tkinter.py` for a complete Tkinter
example that embeds asyncio inside ``tkinter.mainloop()`` using
:func:`start_guest_run`.

.. seealso::

   The `asyncio-guest <https://github.com/congzhangzh/asyncio-guest>`__
   project — the proof of concept this feature is based on — has runnable
   examples for many more hosts: Tkinter, Qt (PyQt5/PySide6), GTK,
   pygame, Win32 and Tornado.

   `Trio's guest mode
   <https://trio.readthedocs.io/en/stable/reference-lowlevel.html#using-guest-mode-to-run-trio-on-top-of-other-event-loops>`__,
   which pioneered this approach.

.. rubric:: API

.. function:: start_guest_run(async_fn, *args, run_sync_soon_threadsafe, done_callback)

   Run *async_fn* as a guest inside another event loop.

   Must be called from the host event loop's thread.  The host loop
   (e.g. ``tkinter.mainloop()``) remains in control of that thread;
   asyncio I/O polling runs in a background non-daemon thread that is
   joined when the run finishes.

   :param async_fn: The async function to run as the top-level coroutine.
   :param args: Positional arguments forwarded to *async_fn*.
   :param run_sync_soon_threadsafe: A callable that schedules a zero-argument
       callable on the host event loop's thread.  It must be thread-safe
       and must not raise.  It must return before the callable runs, even
       when called on the host thread, and must never wait for the
       callable to run, which can deadlock.  It is called both on the host
       thread, where it must not wait for the host loop at all, and from
       the I/O thread, where it may wait for the host loop to accept the
       callable, but then the run, and the interpreter's exit, hang if the
       host loop stops while it waits.  It need not preserve ordering.  If
       it raises when :func:`start_guest_run` first calls it, without
       having run the callable, the run's tasks are cancelled and
       awaited, the loop is closed, and the exception propagates.  For
       Tkinter, see the example above: every tkinter call made from
       another thread waits for Tk's main loop to accept it, and its
       ``TkHost`` class explains what that means.  For Qt, use a queued
       connection, not a direct, automatic or blocking one; etc.
   :param done_callback: Called on the host thread after the run has fully
       finished and the loop is closed (see :ref:`asyncio-guest-lifecycle`).
       Receives the :class:`Task` as its sole argument.  Inspect the
       outcome with :meth:`Task.result`, :meth:`Task.exception`, or
       :meth:`Task.cancelled`.
   :returns: The :class:`Task` wrapping *async_fn*.

   To cancel the task from the host, use::

       loop.call_soon_threadsafe(task.cancel)

   This wakes the I/O thread from its selector wait so cancellation is
   processed promptly.

   .. versionadded:: 3.16

.. _asyncio-guest-lifecycle:

Lifecycle and Cleanup
=====================

The main task starts running in the first callback passed to
*run_sync_soon_threadsafe*, after :func:`start_guest_run` has returned,
unless the loop has an eager task factory (see :func:`eager_task_factory`),
for example set by *async_fn*, which runs the main task's first step at
once.  If the start then fails, the cancelled tasks' cleanup runs inside
:func:`start_guest_run`, where it must not wait for the host.

The guest loop is the host thread's running loop only while guest code
runs, such as in batches of callbacks, at start, and in the final cleanup.
Around each batch of callbacks, the thread's running loop, asynchronous
generator hooks (see :func:`sys.set_asyncgen_hooks`) and coroutine
origin tracking depth are switched to the guest's, and the host's are
restored afterwards.  So inside guest tasks :func:`get_running_loop`
returns the guest loop and starting another event loop — including a
nested :func:`asyncio.run` or :meth:`loop.run_until_complete` — raises
:exc:`RuntimeError`, while host code keeps its own state: asynchronous
generators it iterates are not finalized by the guest loop, and the host
may itself be an asyncio event loop.  :meth:`loop.is_running()
<asyncio.loop.is_running>` returns ``True`` for the whole guest run.

When the main task finishes, cleanup equivalent to :func:`asyncio.run`
takes place on the host thread: remaining tasks are cancelled,
asynchronous generators and the default executor are shut down, the I/O
thread is joined, and the loop is closed.  Only then is *done_callback*
invoked.

If the interpreter exits while a guest run is unfinished, the run is
abandoned: the I/O thread is woken and joined so that interpreter
shutdown does not hang, pending tasks are not cancelled, and
*done_callback* is not called.

Signal Handling
===============

In guest mode the *host* owns signal handling:

* The guest loop never touches :func:`signal.set_wakeup_fd`, neither to
  install a file descriptor nor to reset it on close, so the host's
  signal wake-up pipeline stays intact.
* :meth:`loop.add_signal_handler` and :meth:`loop.remove_signal_handler`
  raise :exc:`RuntimeError`.
* To let asyncio code react to a signal, catch it in the host (with
  :func:`signal.signal` or the host framework's facilities) and forward
  it into the loop with :meth:`loop.call_soon_threadsafe`.

Host Requirements
=================

* *run_sync_soon_threadsafe* must be thread-safe, must not raise, must
  return before the callable runs, and must never wait for the callable
  to run.  On the host thread it must not wait for the host loop at all.
  It may run callbacks in any order.
* Host code running *outside* guest callbacks (for example a GUI button
  handler) must interact with the loop exclusively through
  :meth:`loop.call_soon_threadsafe`, even though it runs on the loop's
  own thread: the I/O thread may be inside the selector, and only
  ``call_soon_threadsafe`` wakes it safely.  Complete a future with
  ``loop.call_soon_threadsafe(fut.set_result, value)``, and create a task
  with ``loop.call_soon_threadsafe(loop.create_task, coro)``.
  :func:`asyncio.create_task` from host code raises :exc:`RuntimeError`,
  as no loop is running, or with an asyncio host, creates the task on the
  host's loop.
* In :ref:`debug mode <asyncio-debug-mode>`, :meth:`loop.call_soon`,
  :meth:`loop.call_later`, :meth:`loop.call_at` and
  :meth:`loop.create_task` raise :exc:`RuntimeError` when host code calls
  them between batches, as they do when called from another thread.  The
  check reports the mistake rather than repairing it, so the operation may
  be left half done: for example, :meth:`loop.create_task` raises once the
  task is made, leaving it pending, and an unclosed :class:`StreamWriter`
  that the garbage collector finalizes during host code reports "Exception
  ignored" and leaves its connection half closed.  It can't see calls that
  don't schedule through those methods, such as :meth:`loop.stop`,
  :meth:`Handle.cancel`, :meth:`Future.add_done_callback` on a pending
  future, and with a selector event loop :meth:`loop.add_reader` and
  :meth:`loop.remove_reader`.  With an eager task factory,
  :meth:`loop.create_task` runs the task's first step at once instead: with
  the C :class:`Task` implementation (the default), that raises
  :exc:`RuntimeError` in any mode, because the loop isn't the running loop,
  while with the pure-Python :class:`Task`, the step runs as host code,
  which the check may not catch.
* :meth:`loop.stop` is not supported in guest mode.
* A modal loop (a modal dialog, or moving or resizing a window on
  Windows) entered by host code keeps the guest run going, as long as it
  runs the host's queue.  Guest code must not enter one: the batch that
  entered it is still running, so the run makes no progress until it
  returns, and if the modal loop runs the host's queue, it may run
  callables the run already posted re-entrantly.  To show a modal dialog
  from guest code, post it to the host with *run_sync_soon_threadsafe*
  and await its answer.

.. rubric:: Low-level Event Loop Methods

:func:`start_guest_run` drives the loop through three low-level methods
-- :meth:`loop.poll_events() <asyncio.loop.poll_events>`,
:meth:`loop.process_events() <asyncio.loop.process_events>`, and
:meth:`loop.process_ready() <asyncio.loop.process_ready>` -- which
decompose a single iteration of the event loop into independently
callable steps.  See :ref:`asyncio-event-loop` for their reference
documentation.
