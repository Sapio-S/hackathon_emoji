import cv2
import dlib
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

predictor_path = "shape_predictor_68_face_landmarks.dat"

# 初始化
predictor = dlib.shape_predictor(predictor_path)

# 初始化dlib人脸检测器
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture('video2.mp4')
# cap = cv2.VideoCapture(0)

def addEye(img, pos, size, rotate):
	eye= Image.open("eye.png")
	eye = eye.resize(size)
	# eye = eye.rotate(rotate)
	img.paste(eye, pos)

def addMouth(img, pos, size):
	mouth = Image.open("mouth.png")
	mouth = mouth.resize(size)
	img.paste(mouth, pos)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
videoWriter = cv2.VideoWriter('xtzres3.mp4', fourcc, 25, (450,450))

while cap.isOpened():
	ok, cv_img = cap.read()
	if not ok:
		break

	img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)  # 转灰
	faces = dlib.full_object_detections()
	rects = detector(img, 0)
	for i in range(len(rects)):
		faces.append(predictor(img, rects[i]))
	finalImg = Image.open("bg2.jpg")
	face_images = dlib.get_face_chips(img, faces, size=500)
	for img in face_images:
		rects = detector(img, 0)
		for i in range(len(rects)):
			# 全部特征点
			landmarks = [[p.x, p.y] for p in predictor(img, rects[i]).parts()]

			leftEyePos = (
			int((landmarks[36][0] + landmarks[39][0]) / 2) - 90, int((landmarks[36][1] + landmarks[39][1]) / 2) - 20)
			rightEyePos = (
			int((landmarks[42][0] + landmarks[45][0]) / 2) - 90, int((landmarks[42][1] + landmarks[45][1]) / 2) - 20)
			leftEyeSize = (
			int(abs(landmarks[39][0] - landmarks[36][0]) * 2), int(abs(landmarks[37][1] - landmarks[41][1]) * 3))
			rightEyeSize = (
			int(abs(landmarks[45][0] - landmarks[42][0]) * 2), int(abs(landmarks[44][1] - landmarks[46][1]) * 3))

			# leftEyePos = (
			# 	int(int((landmarks[36][0] + landmarks[39][0]) / 2) / 30) * 30 - 90,
			# 	int(int((landmarks[36][1] + landmarks[39][1]) / 2) / 30) * 30 - 20)
			# rightEyePos = (
			# 	int(int((landmarks[42][0] + landmarks[45][0]) / 2) / 30) * 30 - 90,
			# 	int(int((landmarks[42][1] + landmarks[45][1]) / 2) / 30) * 30 - 20)
			# leftEyeSize = (
			# 	int(int(abs(landmarks[39][0] - landmarks[36][0]) * 2) / 10) * 10 + 1,
			# 	int(int(abs(landmarks[37][1] - landmarks[41][1]) * 3) / 10) * 10 + 1)
			# rightEyeSize = (
			# 	int(int(abs(landmarks[45][0] - landmarks[42][0]) * 2) / 10) * 10 + 1,
			# 	int(int(abs(landmarks[45][1] - landmarks[42][1]) * 3) / 10) * 10 + 1)
			# theta = math.tanh(
			# 	(landmarks[39][1] - landmarks[42][1]) / (landmarks[39][0] - landmarks[42][0])) * 180 / 3.14
			# print(theta)
			# if leftEyeSize[0] > 2 * leftEyeSize[1]:
			# 	leftEyeSize = (leftEyeSize[0],  int(leftEyeSize[0]/2))
			addEye(finalImg, leftEyePos, leftEyeSize, 0)
			addEye(finalImg, rightEyePos, rightEyeSize, 0)
			# mouthSize = (int(int(abs(landmarks[63][0] - landmarks[61][0])) / 10) * 10 + 1,
			# 			 int(int(abs(landmarks[61][1] - landmarks[67][1]) * 0.7) / 5) * 5 + 1)
			# mouthPos = (int(landmarks[61][0] / 30) * 30 - 20, int(landmarks[61][1] / 30) * 30 - 85)
			mouthSize = (int(abs(landmarks[63][0] - landmarks[61][0]))+1, int(abs(landmarks[61][1]-landmarks[67][1])*0.7)+1)
			mouthPos = (int(landmarks[61][0]) -20, int(landmarks[61][1]) - 85)
			addMouth(finalImg, mouthPos, mouthSize)

	frame = cv2.cvtColor(np.asarray(finalImg), cv2.COLOR_RGB2BGR)
	videoWriter.write(frame)
videoWriter.release()
cap.release()