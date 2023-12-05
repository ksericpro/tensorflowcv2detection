import tensorflow_hub as hub
import cv2
import numpy
import tensorflow as tf
import pandas as pd
import sys
from config import local as _config 
from random import randint

# constants
print(tf.__version__)
width = 1024
height = 768
model_uri = "./models/efficientdet/d0/1"
label_file = _config.label_file

def service_shutdown(signum, frame):
    print('Caught signal %d' % signum)
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == '__main__':

    #https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1
    print("Step 1> Loading TF model from {}".format(model_uri))
    detector = hub.load(model_uri)

    print("Step 2> Reading Labels from {}".format(label_file))
    labels = pd.read_csv(label_file,sep=';',index_col='ID')
    labels = labels['OBJECT (2017 REL.)']

    print("-labels available-", labels)

    print("Step 3> Getting Videocam handler")

    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX

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

        boxes = detector_output["detection_boxes"]
        scores = detector_output["detection_scores"]
        classes = detector_output["detection_classes"]
        num_detections = detector_output["num_detections"]

        # Processing outputs
        pred_labels = classes.numpy().astype('int')[0] 
        pred_labels = [labels[i] for i in pred_labels]
        #pred_boxes = boxes.numpy()[0].astype('int')
        pred_boxes = boxes.numpy()[0]
        pred_scores = scores.numpy()[0]
        
        for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        # print("score={}".format(score))
            if score < 0.5:
                continue
                
            score_txt = f'{numpy.round(100 * score, decimals=2)}%'
            _xmin= numpy.round(xmin*width).astype('int')
            _ymax = numpy.round(ymax*height).astype('int')
            _xmax = numpy.round(xmax*width).astype('int')
            _ymin = numpy.round(ymin*height).astype('int')
            print("{}:[{}] Drawing rectangle ({}, {}), ({}, {})".format(label, score_txt, _xmin, _ymax, _xmax, _ymin))
            #random_color=list(numpy.random.choice(range(255),size=3))
            #print("A Random color is:",random_color)
            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            img_boxes = cv2.rectangle(rgb,(_xmin, _ymax),(_xmax, _ymin), (r, g, b), 2) 
            cv2.putText(img_boxes, label + "("+score_txt+")", (_xmin, _ymax-10), font, 0.6, (r,g,b), 1, cv2.LINE_AA)

        #Display the resulting frame
        cv2.imshow('webcam',img_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
