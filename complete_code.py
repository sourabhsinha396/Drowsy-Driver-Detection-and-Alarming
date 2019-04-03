import cv2
import dlib
import numpy as np
from threading import Thread
from playsound import playsound


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_left(landmarks):
    top_left_pts = []
    for i in range(37,39):
        top_left_pts.append(landmarks[i])
    
    top_left_mean = np.mean(top_left_pts, axis=0)
    #print(top_left_mean[:,1])
    return int(top_left_mean[:,1])

def bottom_left(landmarks):
    bottom_left_pts = []
    for i in range(40,42):
        bottom_left_pts.append(landmarks[i])
    
    bottom_left_mean = np.mean(bottom_left_pts, axis=0)
    return int(bottom_left_mean[:,1])

def eye_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_left_center = top_left(landmarks)
    bottom_left_center = bottom_left(landmarks)
    eye_distance = abs(top_left_center - bottom_left_center)
    return image_with_landmarks, eye_distance

    

cap = cv2.VideoCapture(0)

counter=0
while True:
    ret, frame = cap.read()   
    image_landmarks, eye_distance = eye_open(frame)
    
    if eye_distance < 8:
        counter=counter+1
        if counter>=6:     #if counted frames> 6 frames it triggers putting of text
            
            cv2.putText(frame, "Subject is Drowsing", (50,450),
                        cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
            output_text = " Drowsiness count: " + str(counter + 1-6)

            cv2.putText(frame, output_text, (50,50),
                        cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
            playsound('C://Users//Asus//Downloads//drowsiness-detection/drowsiness-detection/alarm.wav')
            
    else:
        counter=0
        
   
    cv2.imshow('Live Landmarks', image_landmarks )
    cv2.imshow('Drowsiness Detection', frame )
    
    if cv2.waitKey(33) == ord('a'):
        break
        
cap.release()
cv2.destroyAllWindows() 
