import cv2
import dlib
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input",
					dest='input',
					type=str,
					default="3.jpeg",
					help="input photo directory")

parser.add_argument("-o", "--output",
					dest='output',
					type=str,
					default='res.jpg',
					help="output photo directory")

args = parser.parse_args()
predictor_model = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()# dlib人脸检测器
predictor = dlib.shape_predictor(predictor_model)

# cv2读取图像
test_img_path = args.input
img = cv2.imread(test_img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 人脸数rects
rects = detector(img, 0)

# faces存储full_object_detection对象
faces = dlib.full_object_detections()
for i in range(len(rects)):
	faces.append(predictor(img,rects[i]))

def addEye(img, pos, size, rotate):
	eye= Image.open("eye.png").convert("RGBA")
	eye = eye.resize(size)
	img.paste(eye, pos)

def addMouth(img, pos, size):
	eye= Image.open("mouth.png")
	eye = eye.resize(size)
	img.paste(eye, pos)

def whatever(img):
	# 取灰度
	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# 人脸数rects（rectangles）
	rects = detector(img_gray, 0)
	for i in range(len(rects)):
		landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
		for idx, point in enumerate(landmarks):
			# 68点的坐标
			pos = (point[0, 0], point[0, 1])
			cv2.circle(img, pos, 3, color=(0, 425, 0))
	cv2.imshow("result", img)
	cv2.waitKey(0)


# 将人脸裁剪至合适大小并居中, size=320*320*3
landmarks = []
face_images = dlib.get_face_chips(img, faces, size=500)
finalImg = Image.open("bg.jpg")
for image in face_images:
	rects = detector(image, 0)
	for i in range(len(rects)):
		# 全部特征点
		landmarks = [[p.x, p.y] for p in predictor(image, rects[i]).parts()]

		leftEyePos = (int((landmarks[36][0] + landmarks[39][0])/2) - 90, int((landmarks[36][1] + landmarks[39][1])/2)-20)
		rightEyePos = (int((landmarks[42][0] + landmarks[45][0])/2) - 90, int((landmarks[42][1] + landmarks[45][1])/2)-20)
		leftEyeSize = (int(abs(landmarks[39][0] - landmarks[36][0])*2), int(abs(landmarks[37][1] - landmarks[41][1])*3))
		rightEyeSize = (int(abs(landmarks[45][0] - landmarks[42][0])*2),  int(abs(landmarks[44][1] - landmarks[46][1])*3))
		addEye(finalImg, leftEyePos, leftEyeSize, 0)
		addEye(finalImg, rightEyePos, rightEyeSize, 0)
		mouthSize = (int(abs(landmarks[63][0] - landmarks[61][0])), int(abs(landmarks[61][1]-landmarks[67][1])*0.7))
		mouthPos = (int(landmarks[61][0]) -20, int(landmarks[61][1]) - 85)
		addMouth(finalImg, mouthPos, mouthSize)

		# cv2.imshow("show", image)
		# cv2.waitKey(0)
		# image2 = np.array(image)
		# print(landmarks[36][0],landmarks[39][0], landmarks[36][1], landmarks[39][1])
		# roi = image[landmarks[36][0]:landmarks[39][0], landmarks[36][1]: landmarks[39][1]]  # 利用切片工具，选出感兴趣roi区域
		# # print(roi)
		# cv2.imshow("show", roi)
		# cv2.waitKey(0)
		#
		# rows, cols, _ = roi.shape  # 保存视频尺寸以备用
		# gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 转灰度
		# gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)  # 高斯滤波一次
		#
		# _, threshold = cv2.threshold(gray_roi, 8, 255, cv2.THRESH_BINARY_INV)  # 二值化，依据需要改变阈值
		# contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 画连通域
		# contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
		#
		# for cnt in contours:
		# 	(x, y, w, h) = cv2.boundingRect(cnt)
		# 	pos = (x + int(w / 2), y + int(h / 2))
		# 	print(pos)
			# cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
			# cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
			# cv2.line(roi, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2)
			# cv2.line(roi, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 2)
			# break
		#
		# cv2.imshow("Roi", roi)
		# cv2.imshow("Threshold", threshold)
		# cv2.waitKey(0)


finalImg.save(args.output)



