from abc import ABC, abstractmethod
import logging
from typing import Any
from ROAR.utilities_module.module import Module
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from collections import deque


class ROARManiaPlanner(Module):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging
        self.logger = logging.getLogger(__name__)
        self.agent = agent
        # TODO: figure out what this should be
        self.curr_goal = None

    def run_in_series(self, scene) -> Any:
        """
        On every step, take in a scene and return one of the lat_errors as the one
        to PID on.
        """
        # Decide between lane or patch first,
        # then sort patches by distance and type and return one of them
        if scene:
            print(scene)
            lanes = scene['lane']
            error_at_top = self.find_error_at(
                lanes["top"],
                error_scaling=[
                    (20, 0.1),
                    (40, 0.4),
                    (60, 0.6),
                    (70, 0.7),
                    (80, 0.8),
                    (100, 0.9),
                    (200, 2)
                ]
            )

            error_at_mid = self.find_error_at(
                lanes["mid"],
                error_scaling=[
                    (20, 0.1),
                    (40, 0.6),
                    (60, 0.7),
                    (80, 0.85),
                    (100, 0.975),
                    (200, 1.5)
                ]
            )

            error_at_bot = self.find_error_at(
                lanes["bot"],
                error_scaling=[
                    (20, 0.1),
                    (40, 0.75),
                    (60, 0.8),
                    (80, 0.9),
                    (100, 0.95),
                    (200, 1)
                ]
            )
            # Decide what the correct format for this is
            if error_at_top is None and error_at_mid is None and error_at_bot is None:
                return None

            # TODO: rewrite this in terms of the new errors format
            error = 0

            errors = [error_at_top, error_at_mid, error_at_bot]

            
            return sum([error for error in errors if error]) / 3

    def find_error_at(self, data, error_scaling) -> Any:
        if data is not None:
            x = data[1]
            error = x - self.agent.center_x
            for e, scale in error_scaling:
                if abs(error) <= e:
                    error = error * scale
                    break
            return error

    def run_in_threaded(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass