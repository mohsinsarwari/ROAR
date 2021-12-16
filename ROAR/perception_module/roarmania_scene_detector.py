from ROAR.perception_module.detector import Detector
import numpy as np
import cv2
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from typing import Optional
from collections import deque


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
		self.run_detect_patches = True

		# what pixel row the ground begins on (roughly)
		self.GROUND_ROW = 6 * increment

		#detect far point between ground row and far bottom
		self.far_bottom = 7 * increment

		# range to detect lane
		self.lane_top = 8 * increment
		self.lane_bottom = self.processing_scale[0]


		# range to detect patch
		self.patch_top = int(6.5 * increment)
		self.patch_bottom = self.processing_scale[0]

		# range to detect backup waypoint also for the further lane detector
		self.backup_top = 6 * increment
		self.backup_bottom = 7 * increment

		# range to detect on patch
		self.in_front = 9 * increment

		#track hsv bounds (captures low wavelength colors (red, yellow, etc.))
		self.track_lowerb=(0, 0, 195)
		self.track_upperb=(165, 255, 255)

		#boost patch hsv bounds
		self.boost_lowerb=(200, 190, 210)
		self.boost_upperb=(255, 255, 255)

		#ice patch hsv bounds
		self.ice_lowerb=(170, 0, 0)
		self.ice_upperb=(255, 255, 150)

		#are we about to go over a patch?
		self.patch_ahead_ice = False
		self.patch_ahead_boost = False

		# deque for smoothening error
		self.error_deqeue = deque(maxlen=3)

	def run_in_series(self, **kwargs):
		scene = dict()
		if self.agent.front_rgb_camera.data is not None:
			image = self.agent.front_rgb_camera.data.copy()

			hsv = self.preprocess(image)

			#section of image to detect lane
			hsv_lane_section = hsv[self.lane_top:self.lane_bottom, :, :]
			#section of image to detect patch
			hsv_patch_section = hsv[self.patch_top:self.patch_bottom, :, :]
			#section of image for backup waypoint
			hsv_backup_section = hsv[self.backup_top:self.backup_bottom, :, :]

			lane_point = self.detect_lane(hsv_lane_section, self.lane_top)
			scene["lane_point"] = lane_point

			backup_lane_point = self.detect_lane(hsv_backup_section, self.backup_top, backup=True)
			scene["backup_lane_point"] = backup_lane_point

			if lane_point is not None:
				image = cv2.circle(image, (lane_point[1], lane_point[0]), 15, (0, 255, 0), -1)

			if backup_lane_point is not None:
				image = cv2.circle(image, (backup_lane_point[1], backup_lane_point[0]), 15, (0, 255, 0), -1)

			if self.run_detect_patches:        

				patches = self.detect_patches(hsv_patch_section, self.patch_top, lane_point)
				scene["patches"] = patches
				for patch in patches:
					if patch[0] == "ice":
						image = cv2.circle(image, (patch[1][1], patch[1][0]), 15, (255, 0, 0), -1)
					else:
						image = cv2.circle(image, (patch[1][1], patch[1][0]), 15, (0, 0, 255), -1)

				hsv_on_patch = hsv[self.in_front:, :, :]
				on_patch = self.detect_on_patch(hsv_on_patch)
				scene["on_patch"] = on_patch

			else:
				scene["patches"] = []
				scene["on_patch"] = False

		print(scene)

		# image = cv2.line(image, (0, self.GROUND_ROW*self.ratio), (image.shape[1], self.GROUND_ROW*self.ratio), [0, 255, 255], 2)
		# image = cv2.line(image, (0, self.far_bottom*self.ratio), (image.shape[1], self.far_bottom*self.ratio), [0, 255, 255], 2)

		#center line vertical
		iimage = cv2.line(image, (image.shape[1] // 2, 0), (image.shape[1] // 2, image.shape[0]), [255, 255, 0], 3) 

		#patch section
		image = cv2.line(image, (0, self.patch_top*self.ratio), (image.shape[1], self.patch_top*self.ratio), [0, 255, 255], 5)
		image = cv2.line(image, (0, self.patch_bottom*self.ratio), (image.shape[1], self.patch_bottom*self.ratio), [0, 255, 255], 5)

		#lane section
		image = cv2.line(image, (0, self.lane_top*self.ratio), (image.shape[1], self.lane_top*self.ratio), [255, 255, 0], 3)
		image = cv2.line(image, (0, self.lane_bottom*self.ratio), (image.shape[1], self.lane_bottom*self.ratio), [255, 255, 0], 3)

		#backup section
		# image = cv2.line(image, (0, self.backup_top*self.ratio), (image.shape[1], self.backup_top*self.ratio), [0, 0, 0], 4)
		# image = cv2.line(image, (0, self.backup_bottom*self.ratio), (image.shape[1], self.backup_bottom*self.ratio), [0, 0, 0], 4)

		#right in front
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
	def detect_lane(self, hsv_main_section, height_offset, backup=False):

		mask = cv2.inRange(src=hsv_main_section, lowerb=self.track_lowerb, upperb=self.track_upperb)
		point = self.find_median(mask, height_offset)

		if backup:
			cv2.imshow("backup", mask)

		if point is not None:
			point_up = point*self.ratio
			return point_up
		else:
			return None

	def scale_error(self, error):
		
		error_scaling=[
			(10, 0.1),
			(20, 0.2),
			(40, 0.3),
			(60, 0.4),
			(80, 0.6),
			(100, 0.9),
			(200, 1)
		]

		# we want small error to be almost ignored, only big errors matter.
		for e, scale in error_scaling:
			if abs(error) <= e:
				error = error * scale

		return error

	# return [((x, y), "patch name"), ...]
	def detect_patches(self, hsv_main_section, height_offset, lane_point):

		patches = []
		points = []

		mask_ice = cv2.inRange(src=hsv_main_section, lowerb=self.ice_lowerb, upperb=self.ice_upperb)
		mask_ice = cv2.bitwise_not(mask_ice)
		#set white border to make sure patches are detected
		mask_ice[:5, :] = 255
		mask_ice[-5:, :] = 255
		mask_ice[:, :5] = 255
		mask_ice[:, -5:] = 255


		mask_boost = cv2.inRange(src=hsv_main_section, lowerb=self.boost_lowerb, upperb=self.boost_upperb)
		mask_boost = cv2.bitwise_not(mask_boost)
		#set white border to make sure patches are detected
		mask_boost[:5, :] = 255
		mask_boost[-5:, :] = 255
		mask_boost[:, :5] = 255
		mask_boost[:, -5:] = 255

		cv2.imshow('boost', mask_boost)

		params = cv2.SimpleBlobDetector_Params() 

		params.filterByInertia = False
		params.filterByConvexity = False

		params.minArea = 200

		params.filterByCircularity = True
		params.minCircularity = 0.10

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
			patches.append(["ice", patch_up])

		for keypoint in keypoints_boost:
			patch = np.array([int(keypoint.pt[1]) + height_offset, int(keypoint.pt[0])])
			patch_up = patch*self.ratio
			patches.append(["boost", patch_up])

		return patches

	def detect_on_patch(self, hsv_on_patch):

		mask_ice = cv2.inRange(src=hsv_on_patch, lowerb=self.ice_lowerb, upperb=self.ice_upperb)
		mask_boost = cv2.inRange(src=hsv_on_patch, lowerb=self.boost_lowerb, upperb=self.boost_upperb)

		cv2.imshow("mask_boost", mask_boost)

		perc_ice = np.sum(mask_ice) / (255*hsv_on_patch.shape[0]*hsv_on_patch.shape[1])
		perc_boost = np.sum(mask_boost) / (255*hsv_on_patch.shape[0]*hsv_on_patch.shape[1])

		patch = None

		if perc_ice > 0.15:
			self.patch_ahead_ice = True
		elif self.patch_ahead_ice:
			self.patch_ahead_ice = False
			patch = "ice"

		if perc_boost > 0.15:
			self.patch_ahead_boost = True
		elif self.patch_ahead_boost:
			self.patch_ahead_boost = False
			patch = "boost"

		return patch




		