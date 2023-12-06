import tensorflow_hub as hub
import cv2
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import signal
import sys
from config import local as _config 
import argparse, os

# constants
width = _config.width
height = _config.height
model_uri = _config.model_uri
label_file = _config.label_file
print(tf.__version__)

def service_shutdown(signum, frame):
    print('Caught signal %d' % signum)
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == '__main__':
    # Get Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default=_config.image)

    args = parser.parse_args()
    image = args.image
    print("image={}".format(image))

    # Register the signal handlers
    signal.signal(signal.SIGTERM, service_shutdown)
    signal.signal(signal.SIGINT, service_shutdown)

    #Load image by Opencv2
    print("Step 1> Reading image file ..{}".format(image))

    if not(os.path.exists(image)):
        print("{} does not exist".format(image))
        os.exit(0)

    img = cv2.imread(image)
    h, w, c = img.shape
    print('width:  ', w)
    print('height: ', h)
    print('channel:', c)

    #Resize to respect the input_shape
    print("Step 2> Resize image to ({}, {})".format(width, height))
    inp = cv2.resize(img, (width , height ))

    #Convert img BGR to RGB
    print("Step 3> Convert from BGR to RGB")
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    # Converting to uint8
    print("Step 4> Convert image to tensor")
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

    # Add dims to rgb_tensor
    print("Step 5> Adding dimensions to tensor")
    print(rgb_tensor.shape)
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    print(rgb_tensor.shape)

    # Apply image detector on a single image.
    print("Step 6> Loading TF model from {}".format(model_uri))
    detector = hub.load(model_uri)

    # Detect the classes in image using model
    print("Step 6> Detect the classes in image using model")
    detector_output = detector(rgb_tensor)
    class_ids = detector_output["detection_classes"]

    print("detector_output: {}".format(detector_output))
    print("detected class: {}".format(class_ids))

    # Creating prediction
    #boxes, scores, classes, num_detections = detector(rgb_tensor)
    print("length={}".format(len(detector_output)))
    #boxes, scores, classes, num_detections, *_ = detector_output
    boxes = detector_output["detection_boxes"]
    scores = detector_output["detection_scores"]
    classes = detector_output["detection_classes"]
    num_detections = detector_output["num_detections"]

    print("classes=",classes)

    print("Step 7> Reading Labels from {}".format(label_file))
    labels = pd.read_csv(label_file,sep=';',index_col='ID')
    labels = labels['OBJECT (2017 REL.)']

    # Processing outputs
    pred_labels = classes.numpy().astype('int')[0] 
    pred_labels = [labels[i] for i in pred_labels]
    #pred_boxes = boxes.numpy()[0].astype('int')
    pred_boxes = boxes.numpy()[0]
    pred_scores = scores.numpy()[0]

    print("pred_labels={}".format(pred_labels))
    print("pred_boxes={}".format(pred_boxes))
    print("pred_scores={}".format(pred_scores))

    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 

    # Putting the boxes and labels on the image
    for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue

        score_txt = f'{numpy.round(100 * score, decimals=2)}%'
        _xmin= numpy.round(xmin*width).astype('int')
        _ymax = numpy.round(ymax*height).astype('int')
        _xmax = numpy.round(xmax*width).astype('int')
        _ymin = numpy.round(ymin*height).astype('int')

        #print("Drawing rectangle (", xmin, ", ", ymax, "), (", xmax,", ", ymin, ")" )
        print("{}:[{}] Drawing rectangle ({}, {}), ({}, {})".format(label, score_txt, _xmin, _ymax, _xmax, _ymin))
        img_boxes = cv2.rectangle(rgb,(_xmin, _ymax),(_xmax, _ymin), (0,255,0),2) 
        cv2.putText(img_boxes, label + "("+score_txt+")", (_xmin, _ymax-10), font, 0.8, (255,255,0), 2, cv2.LINE_AA)
        #cv2.putText(img_boxes, score_txt, (_xmax, _ymax-10), font, 1.0, (255,0,0), 2, cv2.LINE_AA)

    plt.figure(figsize=(10,10))
    plt.imshow(img_boxes)
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
    #input("Press Enter to continue...")
    # When everything done, release the capture
        cv2.destroyAllWindows()