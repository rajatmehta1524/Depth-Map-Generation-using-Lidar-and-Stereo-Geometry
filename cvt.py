import cv2
import os

path = "/home/aniket/NEU_Courses/PRCV/Project/CV_project/test_depth_completion_anonymous/image"
os.chdir(path)
files = os.listdir()
files.remove("jpeg")

for file in files:
	# Load .png image
	print(file[:-4])
	print(path+"/"+file)
	image = cv2.imread(path+"/"+file)

	# Save .jpg image
	cv2.imwrite(path+"/jpeg/"+file[:-4]+".jpeg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
