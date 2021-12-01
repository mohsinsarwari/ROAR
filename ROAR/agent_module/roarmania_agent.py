from ROAR.agent_module.agent import Agent
from pathlib import Path
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.perception_module.scene_detector import SceneDetector
import logging


class RoarmaniaAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.scene_detector = SceneDetector(self)

    def run_step(self, vehicle: Vehicle, sensors_data: SensorsData) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)

        scene = self.scene_detector.run_in_series()

        print(scene)

        return VehicleControl()