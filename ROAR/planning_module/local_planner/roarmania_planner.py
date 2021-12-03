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
        if 'lane' in scene.keys():
            error_list = []
            lane = scene['lane']
            errors = {}
            for section, coord in lane.items():
                if coord:
                    # sections are "top", "mid", "bot"
                    errors[section] = self.find_error_at(coord,
                                                         error_scaling=[
                                                             (20, 0.1),
                                                             (40, 0.4),
                                                             (60, 0.6),
                                                             (70, 0.7),
                                                             (80, 0.8),
                                                             (100, 0.9),
                                                             (116, 1.2)
                                                         ])

            # Decide what the correct format for this is
            if errors["top"][1] == 0 and errors["mid"][1] == 0 and errors["bot"][1] == 0:
                return None

            # TODO: rewrite this in terms of the new errors format
            error = 0
            if error_list[2][1] != 0:
                error = error_list[2][1]
            if error_list[1][1] != 0:
                error = error_list[1][1]
            if error_list[0][1] != 0:
                error = error_list[0][1]
            print(error)
        else:
            return VehicleControl()

    def run_in_threaded(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass