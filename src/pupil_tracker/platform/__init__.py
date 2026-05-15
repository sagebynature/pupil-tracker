"""Platform-specific helpers."""

from pupil_tracker.platform.macos_windows import (
    candidate_at_point,
    list_visible_windows,
    visible_window_candidates,
)
from pupil_tracker.platform.window_activation import activate_window_candidate

__all__ = [
    "activate_window_candidate",
    "candidate_at_point",
    "list_visible_windows",
    "visible_window_candidates",
]
