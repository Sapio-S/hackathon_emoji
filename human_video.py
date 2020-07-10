import cv2
import dlib
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input",
					dest='input',
					type=str,
					default="video2.mp4",
					help="input photo directory. Recommended type: mp4")

parser.add_argument("-o", "--output",
					dest='output',
					type=str,
					default='videores.avi',
					help="output photo directory. Recommended type: avi")

args = parser.parse_args()

predictor_path = "shape_predictor_68_face_landmarks.dat"

# 初始化
predictor = dlib.shape_predictor(predictor_path)

# 初始化dlib人脸检测器
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(args.input)
# cap = cv2.VideoCapture(0)

def transparent_back(img):
	img = img.convert('RGBA')
	L, H = img.size
	color_0 = img.getpixel((0,0))
	for h in range(H):
		for l in range(L):
			dot = (l,h)
			color_1 = img.getpixel(dot)
			if color_1 == color_0:
				color_1 = color_1[:-1] + (0,)
				img.putpixel(dot,color_1)
	return img

def addEye(img, pos, size, rotate):
	eye= Image.open("eye.png").convert('RGBA')
	eye = eye.resize(size)
	img.paste(eye, pos)

def addMouth(img, pos, size):
	mouth = Image.open("mouth.png")
	mouth = mouth.resize(size)
	img.paste(mouth, pos)

def addPupil(img,pos,eyesize):
	pupil=Image.open("pu.png").convert('RGBA')
	pupil=pupil.resize((eyesize[1]-20,eyesize[1]-20))
	# img1 = Image.open(op.join(path, 'img1.png')).convert('RGBA')
	# img2 = Image.open(op.join(path, 'img2.png')).convert('RGBA')
	r, g, b, a = pupil.split()
	img.paste(pupil, pos, mask=a)
	# pupil=pupil.resize((eyesize[1]-20,eyesize[1]-20))
	# img.paste(pupil,pos)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter(args.output, fourcc, 20, (450,450))

lastLeftPos = (0,0)
lastRightPos= (0,0)
lastLeftSize = (0,0)
lastRightSize = (0,0)
while cap.isOpened():
	ok, cv_img = cap.read()
	if not ok:
		break

	img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)  # 转灰
	faces = dlib.full_object_detections()
	rects = detector(img, 0)
	for i in range(len(rects)):
		faces.append(predictor(img, rects[i]))
	finalImg = Image.open("bg.jpg")
	face_images = dlib.get_face_chips(img, faces, size=500)
	for img in face_images:
		rects = detector(img, 0)
		for i in range(len(rects)):
			# 全部特征点
			landmarks = [[p.x, p.y] for p in predictor(img, rects[i]).parts()]
			leftEyePos = (
				int((landmarks[36][0] + landmarks[39][0]) / 2) - 90,
				int((landmarks[36][1] + landmarks[39][1]) / 2) - 20)
			rightEyePos = (
				int((landmarks[42][0] + landmarks[45][0]) / 2) - 90,
				int((landmarks[42][1] + landmarks[45][1]) / 2) - 20)
			
			# 防抖
			if abs(lastLeftPos[0] - leftEyePos[0]) < 5 and abs(lastLeftPos[1] - leftEyePos[1]) < 3:
				leftEyePos = lastLeftPos
			else:
				lastLeftPos = leftEyePos
			if abs(lastRightPos[0] - rightEyePos[0]) < 5 and abs(lastRightPos[1] - rightEyePos[1]) < 3:
				rightEyePos = lastRightPos
			else:
				lastRightPos = rightEyePos

			leftEyeSize = (
				int(abs(landmarks[39][0] - landmarks[36][0]) * 2+1), int(abs(landmarks[37][1] - landmarks[41][1]) * 3)+1)
			rightEyeSize = (
				int(abs(landmarks[45][0] - landmarks[42][0]) * 2+1), int(abs(landmarks[44][1] - landmarks[46][1]) * 3+1))

			# 防抖
			if abs(lastLeftSize[0] - leftEyeSize[0]) < 5 and abs(lastLeftSize[1] - leftEyeSize[1]) < 5:
				leftEyeSize = lastLeftSize
			else:
				lastLeftSize = leftEyeSize
			if abs(lastRightSize[0] - rightEyeSize[0]) < 5 and abs(lastRightSize[1] - rightEyeSize[1]) < 5:
				rightEyeSize = lastRightSize
			else:
				lastRightSize = rightEyeSize
				
			leftEyePos=(leftEyePos[0],leftEyePos[1]+20-int(leftEyeSize[1]*0.2))
			rightEyePos=(rightEyePos[0],rightEyePos[1]+20-int(rightEyeSize[1]*0.2))


			addEye(finalImg, leftEyePos, leftEyeSize, 0)
			addEye(finalImg, rightEyePos, rightEyeSize, 0)
			mouthSize = (int(abs(landmarks[63][0] - landmarks[61][0])+1), int(abs(landmarks[61][1] - landmarks[67][1]) * 0.7)+2)
			mouthPos = (int(landmarks[61][0]) - 20, int(landmarks[61][1]) - 85)
			addMouth(finalImg, mouthPos, mouthSize)
			finalImg.convert('RGBA')


	frame = cv2.cvtColor(np.asarray(finalImg), cv2.COLOR_RGB2BGR)

	videoWriter.write(frame)

videoWriter.release()
cap.release()