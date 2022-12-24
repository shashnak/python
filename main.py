import cv2
import dlib
import numpy as np

img = cv2.imread('assets/kid1.jpg')
img = cv2.resize(img, (720,640))
frame = img.copy()

#____________model for age detection__________"
age_weights = "models/age_deploy.prototxt"
age_config = "models/age_net.caffemodel"
age_net = cv2.dnn.readnet(age_config, age_weight)

 # model requirements for image
agelist  = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

# storing the image dimensions
fh = img.shape[0]
fw = img.shape[1]

boxes = []  # to store the face coordinates
mssg = 'face detected' #to display on image

#__________model for face detection_______#
face_detector = dlib.get_frontal_face_detector()
# converting to grayscale
img_gray = cv2.cvtcolor(frame, cv2.COLOR_BGR2GRAY)

#__________detecting the faces________#
faces = face_detector(img_gray)

# if no faces our detected
if not faces:
    mssg = 'no face detected'
    cv2.putText(img, f'{mssg}', (40,40),
             cv2.FONT_HERSHEY_SIMPLEX, 2,(200), 2)
cv2.imshow('age detected', img)
cv2.waitkey(0)
 else:
#_______________Bonding Face____________#
for face in faces:
           x = face.left()   #extreacting the face coordinates
           y = face.top()
           x2 = face.right()
           y2 = face.bottom()

# rescaling those coordinates for our image
box = [x,y,x2,y2]
Boxes.append(box)
cv2.rectangle(frame, (x,y),(x2,y2)),(00,200,200), 2)

for box in boxes:
    face = frame[box[1]:box[3], box[0]:box[2]]

#_______Image preprocessing______#
    blob = cv2.dnn.blobfromimage(face,1.0,(227,277), model_mean, swapRB=false)

#________age prediction______#
    age_net.setinput(blob)
    agelist_preds = age_net.forward()
    age = agelist[age_preds[0].argmax()]
    cv2.puttext(frame,f'{mssg}:{age}',(box[0],box[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255,255), 2, cv2.LINE_AA)
cv2.imshow("detecting age",frame)
cv2.waitkey(0)


