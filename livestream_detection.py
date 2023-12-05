import tensorflow_hub as hub
import cv2
import numpy
import tensorflow as tf
import pandas as pd

# constants
print(tf.__version__)
width = 512
height = 512
#model_uri = "https://www.kaggle.com/models/tensorflow/efficientdet/frameworks/TensorFlow2/variations/d0/versions/1"
model_uri = "./models/efficientdet/d0/1"
label_file = "labels.csv"

# Carregar modelos

#https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1
print("Step 1> Loading TF model from {}".format(model_uri))

detector = hub.load(model_uri)
#detector = hub.load("efficientdet_lite2_detection_1")

print("Step 2> Reading Labels from {}".format(label_file))
labels = pd.read_csv(label_file,sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']

print("-labels available-", labels)

print("Step 3> Getting Videocam handler")

cap = cv2.VideoCapture(0)

while(True):
    #Capture frame-by-frame
    ret, frame = cap.read()
    
    #Resize to respect the input_shape
    inp = cv2.resize(frame, (width , height ))

    #Convert img to RGB
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    #Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

    #Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    
    #boxes, scores, classes, num_detections = detector(rgb_tensor)
    detector_output = detector(rgb_tensor)
    boxes = detector_output["raw_detection_boxes"]
    scores = detector_output["detection_multiclass_scores"]
    classes = detector_output["detection_classes"]
    
    pred_labels = classes.numpy().astype('int')[0]
    
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    print("pred_labels={}".format(pred_labels))
    #loop throughout the faces detected and place a box around it
    
    for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
       # print("score={}".format(score))
        if score.all() < 0.5:
            continue
            
        score_txt = f'{100 * numpy.round(score,0)}'
        img_boxes = cv2.rectangle(rgb,(xmin, ymax),(xmax, ymin),(0,255,0),1)      
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_boxes,label,(xmin, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
        cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)



    #Display the resulting frame
    cv2.imshow('Result',img_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
