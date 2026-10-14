import time
import queue
import threading
import logging
from typing import Callable, Any, Dict, List, Optional
from core.data_quality import now_ist, format_ist

logger = logging.getLogger("ExecutionQueue")


class ExecutionQueueManager:
    """
    Centralized, thread-safe execution queue for QuantDesk.
    Prevents race conditions, SQLite lock collisions, and progress-bar overwrites:
    1. Runs tasks sequentially in a dedicated background worker thread.
    2. De-duplicates recurring scheduled tasks (e.g. 5m price stream will not pile up).
    3. Exposes queue status and telemetry so the UI always knows what is running and what is waiting.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self._current_task: Optional[Dict[str, Any]] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_worker()

    def _start_worker(self):
        with self._lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._stop_event.clear()
                self._worker_thread = threading.Thread(
                    target=self._worker_loop,
                    name="GlobalExecutionQueueWorker",
                    daemon=True
                )
                self._worker_thread.start()
                logger.info("✓ Global Execution Queue Worker started.")

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            task_id = task.get("id")
            task_name = task.get("name", "Unknown Task")
            task_fn = task.get("fn")
            args = task.get("args", ())
            kwargs = task.get("kwargs", {})
            done_event = task.get("done_event")
            result_holder = task.get("result_holder")

            start_t = time.time()
            with self._lock:
                self._current_task = {
                    "id": task_id,
                    "name": task_name,
                    "type": task.get("type", "GENERIC"),
                    "started_at": format_ist(),
                    "started_time": start_t
                }
            logger.info(f"▶️ [Queue Worker] Starting task: {task_name} (ID: {task_id})")

            try:
                if callable(task_fn):
                    res = task_fn(*args, **kwargs)
                    if result_holder is not None:
                        result_holder["result"] = res
                        result_holder["status"] = "SUCCESS"
            except Exception as e:
                logger.error(f"❌ [Queue Worker] Task failed ({task_name}): {e}", exc_info=True)
                if result_holder is not None:
                    result_holder["error"] = str(e)
                    result_holder["status"] = "FAILED"
            finally:
                elapsed = round(time.time() - start_t, 2)
                logger.info(f"⏹️ [Queue Worker] Finished task: {task_name} in {elapsed}s")
                with self._lock:
                    self._current_task = None
                if done_event is not None:
                    done_event.set()
                self._queue.task_done()

    def submit_task(
        self,
        task_fn: Callable,
        task_name: str,
        task_type: str = "GENERIC",
        args: tuple = (),
        kwargs: dict = None,
        deduplicate: bool = True
    ) -> Dict[str, Any]:
        """
        Submits an asynchronous task to the queue.
        If deduplicate=True, skips if a task of the same task_type is already waiting in queue.
        """
        if kwargs is None:
            kwargs = {}

        self._start_worker()

        with self._lock:
            # Deduplication check: if a task of same type is already waiting, don't pile up
            if deduplicate and task_type != "GENERIC":
                pending = self.get_queued_task_types()
                if task_type in pending:
                    logger.debug(f"⏭️ [Queue] Task '{task_name}' ({task_type}) already queued; skipping duplicate.")
                    return {
                        "status": "DEDUPLICATED",
                        "message": f"Task '{task_name}' already waiting in queue.",
                        "task_type": task_type
                    }

            task_id = f"TASK-{int(time.time() * 1000)}-{id(task_fn) % 1000}"
            task_item = {
                "id": task_id,
                "name": task_name,
                "type": task_type,
                "fn": task_fn,
                "args": args,
                "kwargs": kwargs,
                "enqueued_at": format_ist(),
                "done_event": None,
                "result_holder": None
            }
            self._queue.put(task_item)
            q_size = self._queue.qsize()
            logger.info(f"📥 [Queue] Enqueued: '{task_name}' ({task_type}). Total queued: {q_size}")
            return {
                "status": "QUEUED",
                "task_id": task_id,
                "queue_position": q_size,
                "message": f"Task '{task_name}' enqueued."
            }

    def submit_and_wait(
        self,
        task_fn: Callable,
        task_name: str,
        task_type: str = "GENERIC",
        args: tuple = (),
        kwargs: dict = None,
        timeout: float = 300.0,
        deduplicate: bool = False
    ) -> Any:
        """
        Submits a task to the queue and synchronously waits for it to complete.
        Useful for scheduled jobs that need to hold until the worker finishes this job.
        """
        if kwargs is None:
            kwargs = {}

        self._start_worker()

        done_event = threading.Event()
        result_holder = {"status": "PENDING", "result": None, "error": None}

        task_id = f"SYNC-TASK-{int(time.time() * 1000)}"
        task_item = {
            "id": task_id,
            "name": task_name,
            "type": task_type,
            "fn": task_fn,
            "args": args,
            "kwargs": kwargs,
            "enqueued_at": format_ist(),
            "done_event": done_event,
            "result_holder": result_holder
        }
        self._queue.put(task_item)
        logger.info(f"⏳ [Queue] Sync task enqueued: '{task_name}'. Waiting for completion...")

        completed = done_event.wait(timeout=timeout)
        if not completed:
            logger.warning(f"⚠️ [Queue] Sync task '{task_name}' timed out after {timeout}s.")
            return None

        if result_holder.get("status") == "FAILED":
            raise RuntimeError(f"Task '{task_name}' failed: {result_holder.get('error')}")

        return result_holder.get("result")

    def get_queued_task_types(self) -> List[str]:
        """Returns list of task types currently in the queue."""
        types = []
        try:
            with self._queue.mutex:
                for item in list(self._queue.queue):
                    types.append(item.get("type", "GENERIC"))
        except Exception:
            pass
        return types

    def get_status(self) -> Dict[str, Any]:
        """Returns live telemetry of the execution queue."""
        with self._lock:
            cur = None
            if self._current_task:
                cur = {
                    "id": self._current_task.get("id"),
                    "name": self._current_task.get("name"),
                    "type": self._current_task.get("type"),
                    "started_at": self._current_task.get("started_at"),
                    "running_seconds": round(time.time() - self._current_task.get("started_time", time.time()), 1)
                }

            queued = []
            try:
                with self._queue.mutex:
                    for item in list(self._queue.queue):
                        queued.append({
                            "id": item.get("id"),
                            "name": item.get("name"),
                            "type": item.get("type"),
                            "enqueued_at": item.get("enqueued_at")
                        })
            except Exception:
                pass

            return {
                "is_running": bool(self._current_task is not None),
                "current_task": cur,
                "queued_count": len(queued),
                "queued_tasks": queued
            }


execution_queue = ExecutionQueueManager()
