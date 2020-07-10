import cv2
import dlib
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input",
					dest='input',
					type=str,
					default="video.mp4",
					help="input photo directory. Recommended type: mp4")

parser.add_argument("-o", "--output",
					dest='output',
					type=str,
					default='dinosaur.avi',
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
	eye= Image.open("dinosaureye1.png").convert('RGBA')
	# eye=eye.convert('RGBA')
	eye = eye.resize(size)
	# eye = eye.rotate(rotate)
	r,g,b,a=eye.split()
	img.paste(eye, pos,mask=a)

def addMouth(img, pos, size):
	mouth = Image.open("dmouth1.png").convert('RGBA')
	# mouth=mouth.convert('RGBA')
	mouth = mouth.resize(size)
	r,g,b,a=mouth.split()
	img.paste(mouth, pos,mask=a)

def addEyebrow(img, pos, size):
	eyebrow = Image.open("eb.png").convert('RGBA')
	# mouth=mouth.convert('RGBA')
	eyebrow = eyebrow.resize(size)
	r,g,b,a=eyebrow.split()
	img.paste(eyebrow, pos,mask=a)

def addPupil(img,pos,eyesize):
	pupil=Image.open("pu.png").convert('RGBA')
	pupil=pupil.resize((eyesize[1]-20,eyesize[1]-20))
	r, g, b, a = pupil.split()
	img.paste(pupil, pos, mask=a)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter(args.output, fourcc, 20, (400,400))

while cap.isOpened():
	ok, cv_img = cap.read()
	if not ok:
		break

	img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)  # 转灰
	faces = dlib.full_object_detections()
	rects = detector(img, 0)
	for i in range(len(rects)):
		faces.append(predictor(img, rects[i]))
	finalImg = Image.open("gd.jpg")
	# finalImg.convert('RGBA')
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
			leftEyeSize = (
				int(abs(landmarks[39][0] - landmarks[36][0])+1), int(abs(landmarks[37][1] - landmarks[41][1])*1.5+1))
			rightEyeSize = (
				int(abs(landmarks[45][0] - landmarks[42][0])+1), int(abs(landmarks[44][1] - landmarks[46][1])*1.5+1))
			# theta = math.tanh(
			# 	(landmarks[39][1] - landmarks[42][1]) / (landmarks[39][0] - landmarks[42][0])) * 180 / 3.14
			# print(theta)
			leftEyePos=(leftEyePos[0]-10,leftEyePos[1]+20-int(leftEyeSize[1]*0.2))
			rightEyePos=(rightEyePos[0]-80,rightEyePos[1]+20-int(rightEyeSize[1]*0.2))
			leftPupilPos=(
				int(leftEyePos[0]+leftEyeSize[0]*(landmarks[37][0]-landmarks[36][0])/(landmarks[39][0]-landmarks[36][0])),
				int((landmarks[37][1] + landmarks[38][1]) / 2) +10
			)
			rightPupilPos=(
				int(rightEyePos[0]+rightEyeSize[0]*(landmarks[43][0]-landmarks[42][0])/(landmarks[45][0]-landmarks[42][0])),
				int((landmarks[43][1] + landmarks[44][1]) / 2) +10
			)
			finalImg.convert('RGBA')
			addEye(finalImg, leftEyePos, leftEyeSize, 0)
			addEye(finalImg, rightEyePos, rightEyeSize, 0)
			mouthSize = (int(abs(landmarks[63][0] - landmarks[61][0])*3+1), int(abs(landmarks[61][1] - landmarks[67][1])*1.2)+8)
			mouthPos = (int(landmarks[61][0] - 120 - mouthSize[0]/8), int(landmarks[61][1]- 115 - mouthSize[1]/3) )
			addMouth(finalImg, mouthPos, mouthSize)
			leftEyebrowPos=(
				int(landmarks[17][0]-10),
				int(landmarks[19][1]+30)
			)
			rightEyebrowPos=(
				int(landmarks[22][0]-100),
				int(landmarks[24][1]+30)
			)
			leftEyebrowSize=(
				int((landmarks[21][0]-landmarks[17][0])*0.3),
				int(((landmarks[17][1]+landmarks[21][1])*0.5-landmarks[19][1])*0.5)
			)
			rightEyebrowSize=(
				int((landmarks[26][0]-landmarks[22][0])*0.3),
				int(((landmarks[26][1]+landmarks[22][1])*0.5-landmarks[24][1])*0.5)
			)
			addEyebrow(finalImg,leftEyebrowPos,leftEyebrowSize)
			addEyebrow(finalImg,rightEyebrowPos,rightEyebrowSize)


	frame = cv2.cvtColor(np.asarray(finalImg), cv2.COLOR_RGB2BGR)

	videoWriter.write(frame)
    
videoWriter.release()
cap.release()