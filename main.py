import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import imutils
import argparse
import os
import math
from PIL import ImageTk, Image

#load the trained model to classify sign
from keras.models import load_model
model = load_model('my_model.h5')

SIGNS = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons',
           44:'Parking',
           45:'No stopping',
           46: 'None' 
        }

CLASS_NUMBER = 46

# Clean all previous files
def clean_images_output():
	file_list = os.listdir('./outputs/')
	for file_name in file_list:
		os.remove('./outputs/' + file_name)

def clean_images_possible():
    file_list_poss = os.listdir('./possible/')
    for file_name in file_list_poss:
        os.remove('./possible/' + file_name)

def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    return image[top:bottom,left:right]

def classify(count):
    global label_packed
    image = Image.open('./possible/possible-sign' + str(count) + '.png')
    image = image.resize((30,30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict([image])[0]
    sign = 45
    index = 0
    lastProb = 0
    for prob in pred:
      if prob>0.71 and prob>=lastProb:
        sign = index 
        lastProb = prob
      index = index + 1
    return sign

def contourIsCircle(perimeter, centroid, threshold):
    # Compute signature of contour
    # threshold taken from a reference paper on signatures for circle detection
    result=[]
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result ]

    # Check signature of contour.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold: # is a sign
        return True
    else:                # is not a sign
        return False

def shape_detection(img):
    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    
    # using a findContours() function
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    i = 0
    
    for contour in contours:
        # here we are ignoring first counter because findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue
        area = cv2.contourArea(contour)
        if area > 60:
            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(co ntour, 0.009 * cv2.arcLength(contour, True), True)
          
            # finding center point of shape
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

            if len(approx) >= 3 and len(approx) <= 6: #checking shapes other then circle
                return True
            else:
                is_sign = contourIsCircle(contour, [x, y], 0.35)
                return is_sign
        return False

def main(args):
	#Clean previous image    
    clean_images_possible()
    clean_images_output()

    #Detection phase
    vidcap = cv2.VideoCapture(args.file_name)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(3)  # float
    height = vidcap.get(4) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, fps , (640,480))

    success = True
    count = 0
    coordinates = []
    while True:
        success, frame = vidcap.read()
        if not success:
            print("FINISHED")
            break
        width = frame.shape[1]
        height = frame.shape[0]
        imageFrame = cv2.resize(frame, (640,480))

        print("Frame:{}".format(count))

        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
        
        # Set range for red color and define mask
        red_lower = np.array([136, 120, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
        
        # Set range for yellow color and define mask
        yellow_lower = np.array([7, 144, 144], np.uint8)
        yellow_upper = np.array([25, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
        
        # Set range for blue color and define mask
        blue_lower = np.array([104,158,127], np.uint8)
        blue_upper = np.array([118,255,255], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
        
        # Morphological Transform, Dilation (to make something wider/larger) for each color and bitwise_and 
        # operator between imageFrame and mask determines to detect only that particular color
        kernal = np.ones((5, 5), "uint8")

        original_image = imageFrame.copy()
        
        # For red color
        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(hsvFrame, hsvFrame, mask = red_mask)
        
        # For yellow color
        yellow_mask = cv2.dilate(yellow_mask, kernal)
        res_yellow = cv2.bitwise_and(hsvFrame, hsvFrame, mask = yellow_mask)
        
        # For blue color
        blue_mask = cv2.dilate(blue_mask, kernal)
        res_blue = cv2.bitwise_and(hsvFrame, hsvFrame, mask = blue_mask)
        
        # Creating contour to track red color
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour) #labelling
                coordinate = [(x,y), (x+w,y+h)]
                image = cropSign(imageFrame, coordinate)
                if image is not None:
                    if (shape_detection(image)):
                        count = count + 1
                        cv2.imwrite('./possible/possible-sign' + str(count) + '.png' , image)
                        sign = classify(count)
                        print(sign)
                        if (sign<=44):
                            cv2.imwrite('./outputs/' + str(count) + '.' + str(sign+1) + '.' + SIGNS[sign+1] + ' red.png', image)
                            cv2.rectangle(original_image, coordinate[0],coordinate[1], (0, 255, 0), 1)
                            font = cv2.FONT_HERSHEY_PLAIN
                            cv2.putText(original_image,SIGNS[sign+1],(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
        
        # Creating contour to track yellow color
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                coordinates = [(x,y), (x+w,y+h)]
                image = cropSign(imageFrame, coordinates)
                if image is not None:
                    if (shape_detection(image)):
                        count = count + 1
                        cv2.imwrite('./possible/possible-sign' + str(count) + '.png' , image)
                        sign = classify(count)
                        if (sign<44):
                            cv2.imwrite('./outputs/' + str(count) + '.' + str(sign+1) + '.' + SIGNS[sign+1] + ' yellow.png', image)
                            print(sign+1)
                            cv2.rectangle(original_image, coordinate[0],coordinate[1], (0, 255, 0), 1)
                            font = cv2.FONT_HERSHEY_PLAIN
                            cv2.putText(original_image,SIGNS[sign+1],(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
                            
        
        # Creating contour to track blue color
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                coordinates = [(x,y), (x+w,y+h)]
                image = cropSign(imageFrame, coordinates)
                if image is not None:
                    if (shape_detection(image)):
                        count = count + 1
                        cv2.imwrite('./possible/possible-sign' + str(count) + '.png' , image)
                        sign = classify(count)
                        if (sign<=44):
                            cv2.imwrite('./outputs/' + str(count) + '.' + str(sign+1) + '.' + SIGNS[sign+1] + ' blue.png', image)
                            print(sign+1)
                            cv2.rectangle(original_image, coordinate[0],coordinate[1], (0, 255, 0), 1)
                            font = cv2.FONT_HERSHEY_PLAIN
                            cv2.putText(original_image,SIGNS[sign+1],(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)

        out.write(original_image)
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NLP Assignment Command Line")
    
    parser.add_argument(
      '--file_name',
      default= "./german_short.mp4",
      help= "Video to be analyzed"
      )

    args = parser.parse_args()
    main(args)