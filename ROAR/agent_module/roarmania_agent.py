from ROAR.agent_module.agent import Agent
from pathlib import Path
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.perception_module.scene_detector import SceneDetector
from ROAR.control_module.real_world_image_based_pid_controller import RealWorldImageBasedPIDController as PIDController
from collections import deque
import logging
import numpy as np
from typing import Optional


class RoarmaniaAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.scene_detector = SceneDetector(self)
        self.GROUND_ROW = 140
        self.center_x = 72
        self.controller = PIDController(agent=self)
        self.prev_steerings: deque = deque(maxlen=10)

    def run_step(self, vehicle: Vehicle, sensors_data: SensorsData) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)

        scene = self.scene_detector.run_in_series()
        print(scene)
        if 'lane' in scene.keys():
            error_list = []
            lane = scene['lane']
            for l in lane:
                # print("l: ", l)
                # print("l[0]: ", l[0])
                error_list.append(self.find_error_at(l,
                                                     error_scaling=[
                                                         (20, 0.1),
                                                         (40, 0.4),
                                                         (60, 0.6),
                                                         (70, 0.7),
                                                         (80, 0.8),
                                                         (100, 0.9),
                                                         (116, 1.2)
                                                     ])
                                  )
            print(error_list)

            if error_list[2][1] == 0 and error_list[1][1] == 0 and error_list[0][1] == 0:
                # did not see the line
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

            error = 0
            if error_list[2][1] != 0:
                error = error_list[2][1]
            if error_list[1][1] != 0:
                error = error_list[1][1]
            if error_list[0][1] != 0:
                error = error_list[0][1]
            print(error)

            self.kwargs["lat_error"] = error
            self.vehicle.control = self.controller.run_in_series()
            print(self.vehicle.control)
            self.prev_steerings.append(self.vehicle.control.steering)

            # self.logger.info(f"line recognized: {error}| control: {self.vehicle.control}")
            return self.vehicle.control
        else:
            return VehicleControl()

    def find_error_at(self, data, error_scaling) -> Optional[tuple]:
        y = 116 - (data[0]/5 - self.GROUND_ROW)
        x = data[1]/5 - self.GROUND_ROW
        error = x - self.center_x
        for e, scale in error_scaling:
            if abs(error) <= e:
                error = error * scale
                break
        return y, error

    def execute_prev_command(self):
        # no lane found, execute the previous control with a decaying factor
        self.logger.info("Executing prev")

        if np.average(self.prev_steerings) < 0:
            self.vehicle.control.steering = -1
        else:
            self.vehicle.control.steering = 1
        # self.logger.info("Cannot see line, executing prev cmd")
        self.prev_steerings.append(self.vehicle.control.steering)
        self.vehicle.control.throttle = 0.18
        self.logger.info(f"No Lane found, executing discounted prev command: {self.vehicle.control}")
        return self.vehicle.control