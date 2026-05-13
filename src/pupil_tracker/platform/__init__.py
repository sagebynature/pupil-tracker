"""Platform-specific helpers."""

from pupil_tracker.platform.macos_windows import (
    candidate_at_point,
    list_visible_windows,
    visible_window_candidates,
)

__all__ = ["candidate_at_point", "list_visible_windows", "visible_window_candidates"]
