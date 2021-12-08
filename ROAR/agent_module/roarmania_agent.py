from ROAR.agent_module.agent import Agent
from pathlib import Path
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.perception_module.roarmania_scene_detector import RoarmaniaSceneDetector
from ROAR.planning_module.local_planner.roarmania_planner import ROARManiaPlanner
from ROAR.control_module.real_world_image_based_pid_controller import RealWorldImageBasedPIDController as PIDController
from collections import deque
import logging
import numpy as np
from typing import Optional


class RoarmaniaAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.scene_detector = RoarmaniaSceneDetector(self)
        self.center_x = 360
        self.controller = PIDController(agent=self)
        self.prev_steerings: deque = deque(maxlen=10)
        self.planner = ROARManiaPlanner(self)

    def run_step(self, vehicle: Vehicle, sensors_data: SensorsData) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)

        if self.front_rgb_camera.data is not None:
            scene = self.scene_detector.run_in_series()
            lat_error = self.planner.run_in_series(scene)
            self.kwargs["on_patch"] = scene["on_patch"]
            print("error: ", lat_error)
            if lat_error is not None:
                self.kwargs["lat_error"] = lat_error
                self.vehicle.control = self.controller.run_in_series()
                print(self.vehicle.control)
                self.prev_steerings.append(self.vehicle.control.steering)

                # self.logger.info(f"line recognized: {error}| control: {self.vehicle.control}")
                return self.vehicle.control
            else:
                # did not see the line
                # TODO: REMOVE THIS
                return VehicleControl()
                neutral = -90
                incline = self.vehicle.transform.rotation.pitch - neutral
                if incline < -10:
                    # is down slope, execute previous command as-is
                    # get the PID for downhill
                    long_control = self.controller.long_pid_control()
                    self.vehicle.control.throttle = long_control
                    return self.vehicle.control

                else:
                    # is flat or up slope, execute adjusted previous command
                    return self.execute_prev_command()
        else:
            return VehicleControl()


    def execute_prev_command(self):
        # no lane found, execute the previous control with a decaying factor
        self.logger.info("Executing prev")

        if np.average(self.prev_steerings) < 0:
            self.vehicle.control.steering = -1
        else:
            self.vehicle.control.steering = 1
        # self.logger.info("Cannot see line, executing prev cmd")
        self.prev_steerings.append(self.vehicle.control.steering)
        self.vehicle.control.throttle = 0.09
        self.logger.info(f"No Lane found, executing discounted prev command: {self.vehicle.control}")
        return self.vehicle.control