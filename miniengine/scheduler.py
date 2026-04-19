"""
Request scheduler — the core orchestrator of the serving engine.

The scheduler sits between the HTTP server and the model engine:

    Server  ──add_request()──▶  Scheduler  ──prefill/decode──▶  Engine
      ▲                            │
      └─── token_queue (stream) ◄──┘

It runs in a background thread, repeatedly calling step() which:
  1. Admits waiting requests and prefills them  (WAITING → RUNNING)
  2. Runs one decode step on every running request
  3. Retires finished requests                    (RUNNING → FINISHED)

"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from miniengine.core import Request, RequestStatus, TokenOutput
from miniengine.engine import Engine

logger = logging.getLogger(__name__)


class Scheduler:
    """
    Continuous-batching FCFS scheduler.
    continuous batching scheduler.

    at each iteration step we fill our batch up to max_running,
    prefill (currently blocking/non batched) them and then
    batch decode all of this. this makes new equests join decode
    in the same step as prefill.

    we also bookeep finished jobs to make sure we can admit new jobs
    later.

    Public API (thread-safe):
        add_request(req)   — enqueue a new request
        start()            — launch the background scheduling loop
        stop()             — gracefully shut down

    Internal (called from the scheduler thread):
        step()             — one full scheduling iteration
    """

    def __init__(
        self,
        engine: Engine,
        max_running: int = 16,
        scheduling_mode: str = "continuous",
    ):
        assert scheduling_mode in ("naive", "static", "continuous"), (
            f'please use a proper mode ("naive", "static", "continuous"'
        )
        self.engine = engine
        self.max_running = max_running
        self.scheduling_mode = scheduling_mode

        # Queues
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        # Thread control
        self._lock = threading.Lock()
        self._running_flag = False
        self._thread: threading.Thread | None = None

        # Stats
        self.total_finished: int = 0
        self.total_generated_tokens: int = 0

    # ── Public API (thread-safe) ────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        """Enqueue a request for scheduling."""
        with self._lock:
            self.waiting.append(request)
            logger.info(
                "Enqueued request %s  (prompt_len=%d, waiting=%d)",
                request.request_id,
                request.num_input_tokens,
                len(self.waiting),
            )

    def start(self) -> None:
        """Start the scheduler loop in a background daemon thread."""
        self._running_flag = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Signal the scheduler to stop and wait for the thread to join."""
        self._running_flag = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("Scheduler stopped")

    # ── Main loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running_flag:
            has_work = bool(self.waiting) or bool(self.running)
            if not has_work:
                time.sleep(0.005)  # idle sleep to avoid busy-waiting
                continue
            try:
                self.step()
            except Exception:
                logger.exception("Scheduler step failed")

    # ── Scheduling step ─────────────────────────────────────────────────

    def step(self) -> list[Request]:
        """
        one scheduling iteration of continuous batching.

        loop stpes:
        1. admit up to max_running and prefill individually
            a. will work on the abtched prefill+decode mixed inference later tiem permitting
        2. decode batched
        3. retire finished requests.

        Returns list of requests that finished in this step.
        """
        finished: list[Request] = []

        # decides based on mode what we can pick
        if self.scheduling_mode == "naive":
            slot_cap = 1
        elif self.scheduling_mode == "static" and self.running:
            slot_cap = 0  # only admit when running is done
        else:
            slot_cap = self.max_running

        admitted: list[Request] = []
        with self._lock: # waiting is shared hence under lock
            while (len(self.running) + len(admitted) < slot_cap) and self.waiting:
                admitted.append(self.waiting.popleft())

        # ── prefill each admitted request individually ───────
        for req in admitted:
            req.status = RequestStatus.RUNNING
        # edge case of EOS at first end of prefil
        # so making sure they do not enter the batched decode below.
            try:
                token_id = self.engine.prefill(req)
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    self.running.append(req)
            except Exception:
                logger.exception("prefill failed for request %s", req.request_id)
                self._fail_request(req, finished)

        # batch call
        if self.running:
            try:
                next_tokens = self.engine.batched_decode(self.running)
                for req, token_id in zip(self.running, next_tokens):
                    req.output_ids.append(token_id)
                    self._stream_token(req, token_id)
            except Exception:
                logger.exception("batched_decode failed; retiring all running requests")
                for req in list(self.running):
                    self._fail_request(req, finished)
                self.running = []
                return finished

        # retire finished stuff
        still_running = []
        for req in self.running:
            if self._check_finished(req, req.output_ids[-1]):
                self._finish_request(req, finished)
            else:
                still_running.append(req)
        self.running = still_running

        return finished

    # ── Helpers ─────────────────────────────────────────────────────────

    def _check_finished(self, req: Request, token_id: int) -> bool:
        """Decide whether a request should stop generating."""
        if req.is_finished:
            return True
        if self.engine.is_stop_token(token_id):
            return True
        return False

    def _stream_token(self, req: Request, token_id: int) -> None:
        """Push a generated token into the request's streaming queue."""
        text = self.engine.decode_token(token_id)
        req.token_queue.put(TokenOutput(token_id=token_id, token_text=text, finished=False))

    def _finish_request(self, req: Request, finished_list: list[Request]) -> None:
        """Mark a request as finished and return its KV pool slot."""
        req.status = RequestStatus.FINISHED
        if req.slot_idx is not None:
            self.engine.free_slot(req.slot_idx)
            req.slot_idx = None
        req.token_queue.put(TokenOutput(token_id=-1, token_text="", finished=True))
        finished_list.append(req)

        self.total_finished += 1
        self.total_generated_tokens += req.num_output_tokens
        logger.info(
            "Finished request %s  (output_len=%d, running=%d, waiting=%d)",
            req.request_id,
            req.num_output_tokens,
            len(self.running),
            len(self.waiting),
        )

    def _fail_request(self, req, finished_list):
        if req in self.running:
            self.running.remove(req)
        if req.slot_idx is not None:
            try:
                self.engine.free_slot(req.slot_idx)
            except Exception:
                logger.exception(
                    "free_slot failed during _fail_request for %s", req.request_id
                )
            req.slot_idx = None
        self._finish_request(req, finished_list)
