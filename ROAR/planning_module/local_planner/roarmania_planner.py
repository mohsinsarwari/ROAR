from abc import ABC, abstractmethod
import logging
from typing import Any
from ROAR.utilities_module.module import Module
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from collections import deque
import numpy as np


class ROARManiaPlanner(Module):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging
        self.logger = logging.getLogger(__name__)
        self.agent = agent
        self.side = "center" # Either "center", "left", "right"
        self.last_error = None


        #norm_error value at which to increase scale
        self.inflection = 0.35


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

        error = None

        if scene["lane_point"] is not None:
            #translate lane point into error for pid
            error = self.point_to_error(scene["lane_point"])
        else:
            error = self.last_error

        # We know where the lane is, and there are patches
        if scene["patches"]:
            scene["patches"].sort(key=lambda patch: patch[1][1]) # patch[1][0] is the y_offset
            print("sorted patches: ", scene["patches"])
            for i, patch in enumerate(scene["patches"]):
                patch_t, patch_point = patch
                y, x = patch_point
                if patch_t == "ice":
                    error = self.avoid(patch_point, error)
                if patch_t == "boost":
                    error = self.pursue(patch_point, error)


        self.last_error = error
        return error
       
    def avoid(self, point, error):
        to_patch = self.point_to_error(point)
        return error + (0.4*(error-to_patch))

    def pursue(self, point, error):
        return 0.5*self.point_to_error(point)

    def point_to_error(self, point):
        #get pixel_offset from center
        pixel_offset =  point[1] - self.agent.center_x

        #normalize to [-1, 1]
        norm_offset = pixel_offset / 360

        #scale to have smaller errors be less significant
        if abs(norm_offset) < self.inflection:
            scaled_error = np.sign(norm_offset)*(1/self.inflection)*(norm_offset**2)
        else:
            scaled_error = norm_offset

        return scaled_error


                    
    def repeat_prev_action(self):
        return None


    def run_in_threaded(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass