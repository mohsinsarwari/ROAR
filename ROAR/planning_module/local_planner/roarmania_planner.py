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
        self.side = "center" # Either "center", "left", "right"

    def run_in_series(self, scene) -> Any:
        """
        Return the error to PID on. 
        """
        # Decide between lane or patch first,
        # then sort patches by distance and type and return one of them
        # scene = {"lane_error": error_lane, "patches": [(type, side, y_offset)], "on_patch": type]}
        # type = ["ice", "boost"], side = ["left", "right", "center"], y_offset = float

        # Algorithm: 
        # 1. Follow main lane if a patch is not present.
        # 2. If patch is present and desirable, go for it and give the correct lat_error to controller
        # 3. After you've gone over patch, return back to main lane as quickly as possible. 
        # 4. If can't see main lane, repeat previous action. 
        # CAVEAT: We don't handle the case that we can see patches but not the lane

        # left has to be positive, right has to be negative
        PATCH_ERRORS = {"left": 100, "right": -100}

        if scene["lane_error"] is None:
            # We don't know where the lane is 
            return None

        elif scene["patches"]:
            # We know where the lane is, and there are patches
            scene["patches"].sort(key=lambda patch: patch[2]) # patch[2] is the y_offset
            for patch in scene["patches"]:
                patch_t, side, y_offset = patch
                if patch_t == "ice" and self.side == side:
                    # Ice detected on the same side we are. Try to avoid
                    if side == "center":
                        # Patch detected in center of lane. Go to left by default
                        self.side = "left"
                        return scene["lane_error"] + PATCH_ERRORS["left"]
                    else:
                        self.side = "center"
                        return scene["lane_error"]
                elif patch_t == "boost":
                    # Boost detected, go for it
                    self.side = side
                    return scene["lane_error"] + PATCH_ERRORS[side]
        else:
            # We can see the lane but not patches, follow the lane
            return scene["lane_error"]

            
                    
    def repeat_prev_action(self):
        return None


    def run_in_threaded(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass