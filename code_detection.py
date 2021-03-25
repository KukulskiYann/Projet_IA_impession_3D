import cv2
import numpy as np

texte_couleur= (255,255,255)
box_couleur = (0,0,255) 
boxes = [] 
confidence = []
classe = []
classes = []
font = cv2.FONT_HERSHEY_PLAIN

n = cv2.dnn.readNet('yolov4-obj_best.weights', 'yolov4-obj.cfg')
with open("obj.names", "r") as f:
    classes = f.read().splitlines()
cap = cv2.VideoCapture('Vantop_20210325_091837.mp4') # video ou image

_, img = cap.read()
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
n.setInput(blob) 

output_layers_names = n.getUnconnectedOutLayersNames()
layerOutputs = n.forward(output_layers_names)

for output in layerOutputs:
    for detection in output:
        scores = detection[5:] 
        classe2= np.argmax(scores) 
        confidence2 = scores[classe2] 
        print(confidence2) 

        if confidence2 > 0.2: 
            centrer_x = int(detection[0]*width)
            centrer_y = int(detection[1]*height)
            largeur = int(detection[2]*width)
            hauteur = int(detection[3]*height)

            x = int(centrer_x - largeur/2) 
            y = int(centrer_y - hauteur/2)

            boxes.append([x, y, largeur, hauteur])
            confidence.append((float(confidence2)))
            classe.append(classe2)

    
    liste = cv2.dnn.NMSBoxes(boxes, confidence, 0.2, 0.4) 
for i in liste.flatten(): 
    x, y, largeur, hauteur = boxes[i] 
    label = str(classes[classe[i]]) 
    confidence2 = str(round(confidence[i], 2))
    cv2.rectangle(img, (x,y), (x+largeur, y+hauteur), box_couleur, 3)
    cv2.putText(img, label + " " + confidence2, (x, y-10), font, 5, (texte_couleur), 4) 
    cap.release()
    cv2.imshow('Image', img)
        
    key = cv2.waitKey(0)
    if key== 27: #   touche échappe
        break 
    
cv2.destroyAllWindows()

