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

        self.actual_scale = [1280, 720]
        self.processing_scale = [256, 144]

        self.ratio = 5

        height = self.processing_scale[0]

        increment = height // 10

        #detect patches as well?
        self.detect_patches = False

        # what pixel row the ground begins on (roughly)
        self.GROUND_ROW = 6 * increment

        # range to detect lane and patches
        self.height_1 = 7 * increment
        self.height_2 = 9 * increment

        # range to detect on patch
        self.in_front = 8 * increment

        #track hsv bounds (captures low wavelength colors (red, yellow, etc.))
        self.track_lowerb=(0, 0, 195)
        self.track_upperb=(165, 255, 255)

        #boost patch hsv bounds
        self.boost_lowerb=(230, 230, 230)
        self.boost_upperb=(255, 255, 255)

        #ice patch hsv bounds
        self.ice_lowerb=(170, 0, 0)
        self.ice_upperb=(255, 255, 93)

        #are we about to go over a patch?
        self.patch_ahead = False

    def run_in_series(self, **kwargs):
        scene = dict()
        if self.agent.front_rgb_camera.data is not None:
            image = self.agent.front_rgb_camera.data.copy()

            hsv = self.preprocess(image)

            #section of image to detect lane and patches
            hsv_main_section = hsv[self.height_1:self.height_2, :, :]

            lane_point, error = self.detect_lane(hsv_main_section, self.height_1)
            scene["lane_error"] = error
            if lane_point is not None:
                image = cv2.circle(image, (lane_point[1], lane_point[0]), 15, (0, 255, 0), -1)

            if self.detect_patches:        

                patches, points = self.detect_patches(hsv_main_section, self.height_1, lane_point)
                scene["patches"] = patches
                for point in points:
                    if point[0] == "ice":
                        image = cv2.circle(image, (point[1][0], point[1][1]), 15, (100, 100, 0), -1)
                    else:
                        image = cv2.circle(image, (point[1][0], point[1][1]), 15, (0, 100, 100), -1)

                hsv_on_patch = hsv[self.in_front:, :, :]
                on_patch = self.detect_on_patch(hsv_on_patch)
                scene["on_patch"] = on_patch

            else:
                scene["patches"] = []
                scene["on_patch"] = False

        print(scene)

        image = cv2.line(image, (0, self.height_1*self.ratio), (image.shape[1], self.height_1*self.ratio), [255, 255, 0], 3)
        image = cv2.line(image, (0, self.height_2*self.ratio), (image.shape[1], self.height_2*self.ratio), [255, 255, 0], 3)

        image = cv2.line(image, (0, self.in_front*self.ratio), (image.shape[1], self.in_front*self.ratio), [100, 100, 100], 3)

        cv2.imshow("image", image)

        return scene

    def preprocess(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        resized_hsv = cv2.resize(image, (144, 256), interpolation = cv2.INTER_AREA)
        return resized_hsv

    def find_median(self, img_sec, height_offset):

        perc = np.sum(img_sec) / (255*img_sec.shape[0]*img_sec.shape[1])

        if perc < 0.05:
            return None

        values = []
        for y in range(img_sec.shape[0]):
            for x in range(img_sec.shape[1]):
                if img_sec[y, x] > 0:
                    values.append([y, x])
        return np.median(values, axis=0).astype(int) + [height_offset, 0]

    # return point on lane in main_section
    def detect_lane(self, hsv_main_section, height_offset):

        mask = cv2.inRange(src=hsv_main_section, lowerb=self.track_lowerb, upperb=self.track_upperb)
        point = self.find_median(mask, height_offset)

        if point is not None:
            point_up = point*self.ratio
            return point_up, point_up[1] - (self.actual_scale[1] // 2)
        else:
            return None, None

    # return [((x, y), "patch name"), ...]
    def detect_patches(self, hsv_main_section, height_offset, lane_point):

        patches = []
        points = []

        mask_ice = cv2.inRange(src=hsv_main_section, lowerb=self.ice_lowerb, upperb=self.ice_upperb)
        mask_ice = cv2.bitwise_not(mask_ice)

        mask_boost = cv2.inRange(src=hsv_main_section, lowerb=self.boost_lowerb, upperb=self.boost_upperb)
        mask_boost = cv2.bitwise_not(mask_boost)

        params = cv2.SimpleBlobDetector_Params() 

        params.filterByInertia = False
        params.filterByConvexity = False

        params.minArea = 100

        params.filterByCircularity = True
        params.minCircularity = 0.01

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else: 
            detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints_ice = detector.detect(mask_ice)
        mask_ice=cv2.bitwise_not(mask_ice)

        keypoints_boost = detector.detect(mask_boost)
        mask_boost=cv2.bitwise_not(mask_boost)

        for keypoint in keypoints_ice:
            patch = np.array([int(keypoint.pt[1]) + height_offset, int(keypoint.pt[0])])
            patch_up = patch*self.ratio
            points.append(["ice", patch_up])
            if lane_point is None:
                patches.append(["ice", "left", patch_up[0]])
            elif (patch_up[1] - lane_point[1]) > 0:
                patches.append(["ice", "left", patch_up[0]])
            else:
                patches.append(["ice", "right", patch_up[0]])

        for keypoint in keypoints_boost:
            patch = np.array([int(keypoint.pt[1]) + height_offset, int(keypoint.pt[0])])
            patch_up = patch*self.ratio
            points.append(["boost", patch_up])
            if lane_point is None:
                patches.append(["boost", "left", patch_up[0]])
            elif (patch_up[1] - lane_point[1]) > 0:
                patches.append(["boost", "left", patch_up[0]])
            else:
                patches.append(["boost", "right", patch_up[0]])

        return patches, points

    def detect_on_patch(self, hsv_on_patch):

        mask_ice = cv2.inRange(src=hsv_on_patch, lowerb=self.ice_lowerb, upperb=self.ice_upperb)
        mask_boost = cv2.inRange(src=hsv_on_patch, lowerb=self.boost_lowerb, upperb=self.boost_upperb)

        perc_ice = np.sum(mask_ice) / (255*hsv_on_patch.shape[0]*hsv_on_patch.shape[1])
        perc_boost = np.sum(mask_ice) / (255*hsv_on_patch.shape[0]*hsv_on_patch.shape[1])

        if perc_ice > 0.15:
            self.patch_ahead = True
            return None
        elif self.patch_ahead:
            self.patch_ahead = False
            return "ice"
        else:
            return None

        if perc_boost > 0.15:
            self.patch_ahead = True
            return None
        elif self.patch_ahead:
            self.patch_ahead = False
            return "boost"
        else:
            return None




        