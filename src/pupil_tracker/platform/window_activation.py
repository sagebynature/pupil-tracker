"""Opt-in platform window activation helpers.

Activation is intentionally separate from visible-window enumeration/scoring so the
normal gaze diagnostics path remains side-effect free.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from pupil_tracker.models import WindowCandidate


def activate_window_candidate(candidate: WindowCandidate) -> None:
    """Bring the candidate's owning macOS application to the foreground.

    CoreGraphics gives us the owner PID for visible windows. AppKit can activate
    that running application without requiring us to synthesize clicks. If the
    candidate lacks a PID, fail loudly so callers can report the focus action as
    unavailable rather than guessing by app title.
    """

    if candidate.process_id is None:
        msg = "window candidate does not include a process id"
        raise RuntimeError(msg)

    appkit = cast(Any, import_module("AppKit"))
    running_app = appkit.NSRunningApplication.runningApplicationWithProcessIdentifier_(
        int(candidate.process_id)
    )
    if running_app is None:
        msg = f"no running application for process id {candidate.process_id}"
        raise RuntimeError(msg)

    activated = bool(
        running_app.activateWithOptions_(appkit.NSApplicationActivateIgnoringOtherApps)
    )
    if not activated:
        msg = f"macOS refused to activate {candidate.app_name}"
        raise RuntimeError(msg)
