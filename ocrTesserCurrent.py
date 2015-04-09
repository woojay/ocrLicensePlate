import cv2
import cv2.cv as cv
import numpy as np
import tesseract
#import argparse

# OPEN TESSERACT
api = tesseract.TessBaseAPI()
api.Init(".","eng",tesseract.OEM_DEFAULT)
api.SetPageSegMode(tesseract.PSM_SINGLE_CHAR)

# OPEN VIDEO
cap = cv2.VideoCapture(0)

while (True):
	
	# READ ONE FRAME
	ret, frame = cap.read()

	# OUTPUT FILE (FOR THIS FRAME)
	out = np.zeros(frame.shape,np.uint8)

	# Convert to Gray and filter
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,35,2)
	thresh = thresh2.copy()
	cv2.imshow('thresh',thresh)

	
	height, width = thresh.shape
	
	# COUNTOUR DETECT AND BUILD A LIST
	# THRESH2 GETS CHANGED AFTER THIS CONTOUR DETECTION
	contours,hierarchy = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# FOR EACH CONTOUR IN THE CURRENT FRAME, ID AND BUILD OUTPUT IMAGE
	for cnt in contours:
#   		if cv2.contourArea(cnt)>0:
        	[x,y,w,h] = cv2.boundingRect(cnt)

		if ((h > 30) and (h < 150)) and (0.7 < h/w < 6):
			if y < 5:
				ybot = y
			else:
				ybot = y-5
			if (y+h) > (height - 5):
				ytop = height
			else:
				ytop = y+h+5
			if x < 5:
				xleft = 0
			else:
				xleft = x-5
			if (x+w) > (width -5):
				xright = width
			else:
				xright = x+w+5

			roi = thresh[ybot:ytop,xleft:xright].copy()

			roismall = cv2.resize(roi, (w*2,h*2))
			hroi, wroi = roismall.shape
#			cv2.imshow('forcus', roismall)

########		Enhance section
			# ENHANCEMENT VARIABLES
			scale = 1
			delta = 0
			ddepth = cv2.CV_16S
			roienhance = roismall.copy()

			### trim the edges
#			cut_offset=23
#			gray=gray[cut_offset:-cut_offset,cut_offset:-cut_offset]

#### convert to gray color
#gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)

			### edge enhancing by Sobeling
			# Gradient-X
			roi_x = cv2.Sobel(roienhance,ddepth,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
#grad_x = cv2.Scharr(gray,ddepth,1,0)

			# Gradient-Y
			roi_y = cv2.Sobel(roienhance,ddepth,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
#grad_y = cv2.Scharr(gray,ddepth,0,1)


			abs_grad_x = cv2.convertScaleAbs(roi_x)   # converting back to uint8
			abs_grad_y = cv2.convertScaleAbs(roi_y)
			roienhance = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)

			### Bluring
			image1 = cv2.medianBlur(roienhance,5) 
			image1[image1 < 50]= 255
			image1 = cv2.GaussianBlur(image1,(3,3),0)     
			color_offset=230
			image1[image1 >= color_offset]= 255  
			image1[image1 < color_offset ] = 0      #black

#			#### Insert White Border
#			offset=3
#			image1=cv2.copyMakeBorder(image1, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=(255,255,255)) 

#			cv2.imshow('forcus', image1)
########		Enhance section

			cv2.imshow('focus', roismall)

			height1, width1 = roismall.shape

			iplimage = cv.CreateImageHeader((width1,height1), cv.IPL_DEPTH_8U,1)
			cv.SetData(iplimage, roismall.tostring(),roismall.dtype.itemsize * (wroi))
			tesseract.SetCvImage(iplimage,api)
			text=api.GetUTF8Text()
			conf=api.MeanTextConf()	
			
			cv2.rectangle(gray,(xleft, ybot),(xright,ytop),(0,255,0),2)
			if (len(text) >0):
				cv2.putText(gray,text.strip(),(x,y+h),0,1,(0,255,0))
				print text.strip(),'=', len(text)
			
	cv2.imshow('CAPTURED',gray)
	cv2.imshow('DETECTED',out)

	cv2.moveWindow('CAPTURED',30,500)
#	cv2.moveWindow('CAPTURED',10,10)
	height, width, depth = frame.shape
	cv2.moveWindow('DETECTED',width + 50,500)
#	cv2.moveWindow('DETECTED',20, 20)

	# GET OUT
	if  cv2.waitKey(1) & 0xFF == ord('q'):
	        break

api.End()
cap.release()
cv2.destroyAllWindows()