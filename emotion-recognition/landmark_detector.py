import numpy as np
import cv2
import dlib
import math

detector = dlib.get_frontal_face_detector() #face detecor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # feature predictor

def get_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray for better accuracy
    face = detector(gray, 1)[0] # take only the first face
    shape = predictor(gray, face) # shape will contain the coordinates of the facial features ranging from 1 - 68
    xlist = [] # Initialize xlist for storing x coordinates
    ylist = [] # Initialize ylist for storing y coordinates
    for i in range(1, 68): 
        cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0), thickness=2) # only for identification purpose
        xlist.append(float(shape.part(i).x)) # store the x coordinates
        ylist.append(float(shape.part(i).y)) # store the y coordinates
        
    ''' Before applying a classifier, we need to normalize the coordinates. It we don't do it,
    the same expression with different coordinates will confuse the classifier.So we will take a common point w.r.t face and calculate 
    coordinates relative to it'''    
   
    xmean = np.mean(xlist) # take the mean of all  x coordinates
    ymean =  np.mean(ylist) # take the mean of all y coordinates
    xdist = [(x-xmean) for x in xlist] # calcuate x coordinate w.r.t to the mean position
    ydist = [(y-ymean) for y in ylist] # calcuate x coordinate w.r.t to the mean position
    
    ''' Next is the case where we have to deal whether the face is tilted or not.If the face is titled, the coordinates 
    will not similar to the original ones and again might cause confusion to the classifier.In to fix it, we have to 
    find the tilted angle and add or subtract it accordingly.It is assumed that the bridge of the nose is straight for 
    most of the humans.By this way, we compare the bridge line with the original line and find out the offset angle.'''    
    
    if xlist[26]==xlist[29]: # 26th index corresponds to the coordinates for the tip of the nose and 29th for the top of the bridge.
        angleoffset=0 # If they both have the same x coordinate, then offset will be zero.
    else:
        angleoffset_rad=int(math.atan((ydist[26] - ydist[29]) / (xdist[26] - xdist[29])))
        angleoffset=int(math.degrees(angleoffset_rad))

    if angleoffset<0:
        angleoffset+=90

    else:
        angleoffset-=90
        
    cv2.circle(img, (int(xmean), int(ymean)), 1, (255, 0, 0), thickness=2)       
    cv2.circle(img, (int(xlist[26]), int(ylist[26])), 1, (0, 0, 255), thickness=2)   
    cv2.circle(img, (int(xlist[29]), int(ylist[29])), 1, (0, 0, 255), thickness=2)
    print(angleoffset)
    
    landmarks = []
    for x, y, w, z in zip(xdist, ydist, xlist, ylist):
        landmarks.append(xdist)
        landmarks.append(ydist)
        meannp = np.asarray((ymean,xmean))
        coornp = np.asarray((z,w))
        dist = np.linalg.norm(coornp-meannp)
        landmarks.append(dist)
        angle_rad = (math.atan((z - ymean) / (w - xmean)))
        angle_degree = math.degrees(angle_rad)
        angle_req = int(angle_degree - angleoffset)
        landmarks_v.append(angle_req)

    if len(face_detections)<1:
        landmarks_v="error"

    return  landmarks_v
    
    return landmarks



img = cv2.imread('hem.png')
landmarks = get_landmarks(img)

    
    

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



