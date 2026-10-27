"""
src/app/voice.py
──────────────────
Robust asynchronous text-to-speech worker using pyttsx3 on Windows SAPI.

Design & Threading Guarantees:
  1. Thread-Affinity: On Windows SAPI5, pyttsx3 MUST be initialized inside the
     same worker thread that executes .say() and .runAndWait(), with CoInitialize().
  2. Bounded Queue: Obsolete posture instructions are discarded so speech never
     lags behind real-time physical adjustments.
  3. Clean Lifecycle: Thread-safe toggle, queue flushing, and shutdown.
  4. Diagnostic Mode: Exposes test_diagnostic() to verify hardware speaker output.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class VoiceFeedback:
    """Thread-safe, non-blocking text-to-speech speaker."""

    def __init__(self, rate: int = 160, volume: float = 0.9, enabled: bool = True):
        self._enabled = enabled
        self._rate = rate
        self._volume = volume

        # Queue contains strings to speak or None as shutdown sentinel
        self._q: queue.Queue[Optional[str]] = queue.Queue(maxsize=3)
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._init_error: Optional[str] = None
        self._speaking_lock = threading.Lock()
        self._is_speaking = False

        # Launch dedicated worker
        self._thread = threading.Thread(target=self._worker, daemon=True, name="YogaMitra-VoiceWorker")
        self._thread.start()

        # Wait briefly for worker initialization
        self._ready_event.wait(timeout=3.0)
        if self._init_error:
            logger.warning("[Voice] TTS initialization failed: %s", self._init_error)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and (self._init_error is None)

    @property
    def is_available(self) -> bool:
        return self._init_error is None

    @property
    def _queue(self) -> queue.Queue[Optional[str]]:
        return self._q

    @property
    def is_speaking(self) -> bool:
        with self._speaking_lock:
            return self._is_speaking

    def toggle(self) -> bool:
        """Toggle voice audio on/off. Flushes any pending queue."""
        self._enabled = not self._enabled
        if not self._enabled:
            self.clear_queue()
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self.clear_queue()

    def clear_queue(self) -> None:
        """Discard any unplayed speech instructions."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def speak(self, message: str, clear_pending: bool = True) -> None:
        """
        Queue a short speech instruction.
        If clear_pending is True, any previously queued sentence that has not yet
        started speaking is dropped to prevent outdated posture cues.
        """
        if not self.enabled or not message.strip():
            return

        if clear_pending:
            self.clear_queue()

        try:
            self._q.put_nowait(message.strip())
        except queue.Full:
            # If full, drop older and insert newest
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(message.strip())
            except queue.Full:
                pass

    def shutdown(self, timeout: float = 1.5) -> None:
        """Gracefully terminate worker thread."""
        self._shutdown = True
        self.clear_queue()
        try:
            self._q.put_nowait(None)  # Sentinel
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    # ── Dedicated Worker Loop ─────────────────────────────────────────────────

    def _worker(self) -> None:
        # Initialize COM on this worker thread
        co_init = False
        try:
            import pythoncom
            pythoncom.CoInitialize()
            co_init = True
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("[Voice] CoInitialize notice: %s", exc)

        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            self._init_error = None
        except Exception as exc:
            self._init_error = str(exc)
            engine = None
        finally:
            self._ready_event.set()

        if engine is None:
            return

        while True:
            try:
                item = self._q.get()
                if item is None:
                    # Shutdown sentinel
                    break

                if not self._enabled:
                    continue

                with self._speaking_lock:
                    self._is_speaking = True

                try:
                    engine.say(item)
                    engine.runAndWait()
                except Exception as exc:
                    logger.warning("[Voice] Speech error on '%s': %s", item, exc)
                finally:
                    with self._speaking_lock:
                        self._is_speaking = False
                    # Brief breath between utterances
                    time.sleep(0.15)

            except Exception as exc:
                logger.error("[Voice] Worker unexpected error: %s", exc)

        # Cleanup COM on exit
        if co_init:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass


def test_diagnostic(sample_text: str = "Welcome to Yoga Mitra. Auditory feedback is online.") -> bool:
    """
    Diagnostic routine to verify audio output hardware and SAPI driver.
    Returns True if diagnostic ran successfully, False otherwise.
    """
    print("\n" + "=" * 60)
    print("  YOGA MITRA — VOICE DIAGNOSTIC")
    print("=" * 60)

    try:
        import pyttsx3
    except ImportError:
        print("[FAIL] pyttsx3 package is not installed.")
        return False

    try:
        import pythoncom
        pythoncom.CoInitialize()
        print("[PASS] Windows pythoncom COM library available.")
    except Exception as e:
        print(f"[NOTE] pythoncom status: {e}")

    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        rate = engine.getProperty("rate")
        volume = engine.getProperty("volume")
        print(f"[PASS] SAPI5 TTS Engine initialized successfully.")
        print(f"       Rate: {rate} wpm, Volume: {volume*100:.0f}%")
        print(f"       Available voices: {len(voices)}")
        for i, v in enumerate(voices[:3]):
            print(f"         {i+1}. {v.name} ({v.languages if hasattr(v, 'languages') else 'default'})")

        print(f"\n[TEST] Speaking test phrase: '{sample_text}'...")
        engine.say(sample_text)
        engine.runAndWait()
        print("[PASS] Audio spoken successfully. If you heard speech, hardware is working.")
        print("=" * 60 + "\n")
        return True
    except Exception as exc:
        print(f"[FAIL] Voice engine diagnostic failed: {exc}")
        print("=" * 60 + "\n")
        return False
