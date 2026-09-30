"""Accuracy-guided discrete multiplier adjustment from paper Figure 2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControllerState:
    grid_index: int
    smoothed_accuracy: float | None = None


@dataclass(frozen=True)
class ControllerEvent:
    raw_accuracy: float
    smoothed_accuracy: float
    decision_accuracy: float
    old_grid_index: int
    new_grid_index: int
    old_multiplier: float
    new_multiplier: float
    action: str
    hit_grid_boundary: bool


def update_smoothed_accuracy(
    previous: float | None, raw: float, smoothing: float
) -> float:
    if not 0.0 <= raw <= 1.0:
        raise ValueError("accuracy must be in [0, 1]")
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    return raw if previous is None else smoothing * previous + (1.0 - smoothing) * raw


def controller_decision(
    *,
    accuracy: float,
    target_accuracy: float,
    tolerance: float,
    current_index: int,
    max_index: int,
) -> tuple[int, str]:
    if accuracy < target_accuracy:
        return min(current_index + 1, max_index), "increase"
    if accuracy > target_accuracy + tolerance:
        return max(current_index - 1, 0), "decrease"
    return current_index, "hold"


class AccuracyGridController:
    def __init__(
        self,
        multiplier_grid: tuple[float, ...],
        initial_grid_index: int,
        *,
        target_accuracy: float,
        tolerance: float,
        smoothing: float,
        use_smoothed_accuracy: bool,
    ) -> None:
        self.grid = multiplier_grid
        self.target_accuracy = target_accuracy
        self.tolerance = tolerance
        self.smoothing = smoothing
        self.use_smoothed_accuracy = use_smoothed_accuracy
        self.state = ControllerState(grid_index=initial_grid_index)

    @property
    def multiplier(self) -> float:
        return self.grid[self.state.grid_index]

    def observe(
        self, raw_accuracy: float, *, allow_update: bool = True
    ) -> ControllerEvent:
        smoothed = update_smoothed_accuracy(
            self.state.smoothed_accuracy, raw_accuracy, self.smoothing
        )
        self.state.smoothed_accuracy = smoothed
        decision_accuracy = smoothed if self.use_smoothed_accuracy else raw_accuracy
        old_index = self.state.grid_index
        if allow_update:
            new_index, action = controller_decision(
                accuracy=decision_accuracy,
                target_accuracy=self.target_accuracy,
                tolerance=self.tolerance,
                current_index=old_index,
                max_index=len(self.grid) - 1,
            )
        else:
            new_index, action = old_index, "warmup"
        self.state.grid_index = new_index
        hit_boundary = allow_update and (
            (action == "increase" and old_index == len(self.grid) - 1)
            or (action == "decrease" and old_index == 0)
        )
        return ControllerEvent(
            raw_accuracy=raw_accuracy,
            smoothed_accuracy=smoothed,
            decision_accuracy=decision_accuracy,
            old_grid_index=old_index,
            new_grid_index=new_index,
            old_multiplier=self.grid[old_index],
            new_multiplier=self.grid[new_index],
            action=action,
            hit_grid_boundary=hit_boundary,
        )
