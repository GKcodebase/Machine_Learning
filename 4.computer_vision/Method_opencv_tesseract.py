"""
    Number plate detecting API
    input: image of car
    output: whether the specifice car is allowed or not
    Using openCV,Tesseract and Mysql
"""
import pytesseract
import numpy as np
#importing opencv library
import cv2
#taking input image
image = cv2.imread('car_3.jpg',1)
#converting image into greyscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#Preproccing befor harr classifier
#gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
gray = cv2.GaussianBlur(gray,(5,5), 5)
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
kernel = np.ones((1, 1), np.uint8)
gray = cv2.dilate(gray, kernel, iterations=1)
gray = cv2.erode(gray, kernel, iterations=1)
#Loading Numberplate harr cascade
number_plate = cv2.CascadeClassifier('haarcascade_vehicle_number-plate.xml')
#Using harr cascade to detect the number plate
numberplate = number_plate.detectMultiScale(gray,1.1,10)
#Showing the input and gray scale image
cv2.imshow('imgage',image)
cv2.imshow('gray_scale_image',gray)
cv2.waitKey( )
cv2.destroyAllWindows()
#find a traingle region around the number plate
for (x, y, w, h) in numberplate:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#Showing the detected image
cv2.imshow('Detectes_imgage',image)
cv2.waitKey( )
cv2.destroyAllWindows()
#Cropping the numberplate
plate = image[y:y+h, x:x+w]
cv2.imshow('cropped_numberplate',plate)
cv2.waitKey( )
cv2.destroyAllWindows()
config = ('-l eng --oem 1 --psm 3')
#preprocessing before pytesseract
plate= cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
#plate = cv2.resize(plate, None, fx=2 , fy=2, interpolation=cv2.INTER_CUBIC)
plate = cv2.GaussianBlur(plate, (5, 5), 2)
#plate = cv2.threshold(plate,255,255,cv2.THRESH_BINARY)
plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#plate = cv2.adaptiveThreshold(plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)


#kernel = np.ones((1, 1), np.uint8)
#plate = cv2.dilate(plate, kernel, iterations=1)
#plate = cv2.erode(plate, kernel, iterations=1)
cv2.imshow('cropped_numberplate_proccesed',plate)
cv2.waitKey( )
cv2.destroyAllWindows()

#Using pytesseract to convert the image into text
test = pytesseract.image_to_string(plate ,lang='eng')
print("The detected number plate is: "+test)

