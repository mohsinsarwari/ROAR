from ROAR.perception_module.detector import Detector
import numpy as np
import cv2
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from typing import Optional


class RoarmaniaSceneDetector(Detector):
    def save(self, **kwargs):
        pass

    def __init__(self, agent: Agent, **kwargs):
        super().__init__(agent, **kwargs)

        # what pixel row the ground begins on (roughly)
        self.GROUND_ROW = 140

        #track hsv bounds (captures low wavelength colors (red, yellow, etc.))
        self.track_lowerb=(0, 0, 195)
        self.track_upperb=(165, 255, 255)

    def run_in_series(self, **kwargs):
        scene = dict()
        if self.agent.front_rgb_camera.data is not None:

            hsv, depth = self.preprocess(self.agent.front_rgb_camera.data, self.agent.front_depth_camera.data)

            lane_points = self.detect_lane(hsv, depth)
            scene["lane"] = lane_points

            #patch_points = detect_patches(hsv, depth)
        return scene

    def preprocess(self, img_data, depth_data):
        hsv = cv2.cvtColor(img_data, cv2.COLOR_BGR2HSV)
        resized_hsv = cv2.resize(img_data, (144, 256), interpolation = cv2.INTER_AREA)

        cropped_hsv = resized_hsv[self.GROUND_ROW:, :, :]
        cropped_depth = depth_data[self.GROUND_ROW:, :]

        return cropped_hsv, cropped_depth

    def find_median(self, img_sec, height_offset, width_offset=0):

        perc = np.sum(img_sec) / (255*img_sec.shape[0]*img_sec.shape[1])

        if perc < 0.03:
            return None

        values = []
        for y in range(img_sec.shape[0]):
            for x in range(img_sec.shape[1]):
                if img_sec[y, x] > 0:
                    values.append([y, x])
        return np.median(values, axis=0).astype(int) + [height_offset, 0]


    # Takes in pixel location, returns point in space
    def pixel_to_world_transform(self, pixel_coords, depth):
        return pixel_coords

    # return next three points on lane
    def detect_lane(self, hsv, depth):
        image = self.agent.front_rgb_camera.data.copy()

        mask = cv2.inRange(src=hsv, lowerb=self.track_lowerb, upperb=self.track_upperb)
        # print("img size:", image.shape)
        # print("mask size", mask.shape)

        height_1 = mask.shape[0] // 6
        height_2 = 3 * height_1

        width_1 = mask.shape[0] // 4

        sec1 = mask[:height_1]
        sec2 = mask[height_1:height_2]
        sec3 = mask[height_2:]

        mask = cv2.line(mask, (0, height_1), (mask.shape[1], height_1), 150, 1)
        mask = cv2.line(mask, (0, height_2), (mask.shape[1], height_2), 180, 1)

        img_height_1 = (height_1 + self.GROUND_ROW)*5
        img_height_2 = (height_2 + self.GROUND_ROW)*5

        image = cv2.line(image, (0, self.GROUND_ROW*5), (image.shape[1], self.GROUND_ROW*5), [255, 255, 0], 5)
        image = cv2.line(image, (0, img_height_1), (image.shape[1], img_height_1), [255, 255, 0], 5)
        image = cv2.line(image, (0, img_height_2), (image.shape[1], img_height_2), [255, 255, 0], 5)   

        wid = 5
        wid_up = 10

        avg1 = self.find_median(sec1, 0)

        if avg1 is not None:
            avg1_up = (avg1+[self.GROUND_ROW, 0])*5
            mask[avg1[0]-wid: avg1[0]+wid, avg1[1]-wid: avg1[1]+wid] = 100
            image[avg1_up[0]-wid_up: avg1_up[0]+wid_up, avg1_up[1]-wid_up: avg1_up[1]+wid_up] = [255, 0, 0]
        else:
            avg1_up = None

        avg2 = self.find_median(sec2, height_1)

        if avg2 is not None:
            avg2_up = (avg2+[self.GROUND_ROW, 0])*5
            mask[avg2[0]-wid: avg2[0]+wid, avg2[1]-wid: avg2[1]+wid] = 100
            image[avg2_up[0]-wid_up: avg2_up[0]+wid_up, avg2_up[1]-wid_up: avg2_up[1]+wid_up] = [255, 0, 0]
        else:
            avg2_up = None

        avg3 = self.find_median(sec3, height_2)

        if avg3 is not None:
            avg3_up = (avg3+[self.GROUND_ROW, 0])*5
            mask[avg3[0]-wid: avg3[0]+wid, avg3[1]-wid: avg3[1]+wid] = 100
            image[avg3_up[0]-wid_up: avg3_up[0]+wid_up, avg3_up[1]-wid_up: avg3_up[1]+wid_up] = [255, 0, 0]
        else:
            avg3_up = None

        #cv2.imshow("mask", mask)
        cv2.imshow("image", image)
        # print(avg1, avg2, avg3)
        return {"top": avg1_up, "mid": avg2_up, "bot": avg3_up}
        # return [avg1, avg2, avg3]


    # return [((x, y), "patch name"), ...]
    def detect_patches(self, hsv, depth):
        return None