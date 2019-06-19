import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.pyplot import imshow
import PIL
import pandas as pd
import scipy.io
import tensorflow as tf
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, ActivityRegularization, Flatten, Dropout, Dense
from keras.models import load_model, Model, Sequential
import scipy.misc
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

%matplotlib inline

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.6):
    """
    Filters YOLO boxes by thresholding on object and class confidence.

    :param box_confidence: tensor of shape (19, 19, 5, 1)
    :param boxes: tensor of shape (19, 19, 5, 4)
    :param box_class_probs: tensor of shape (19, 19, 5, 80)
    :param threshold: threshold value to keep the box, here - 0.6

    :return: scores - tensor of shape (None, ) containing the class probability of the selected boxes.
             boxes - tensofrof shape (None, 4) containing (bx, by, bh, bw) coordinates of selected boxes.
             classes - tensor of shape (None, ) containing the index of the class detected by the selected boxes.
    None - because one doesn;t want the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10, ) if there were 10 boxes.
    """

    box_scores = np.multiply(box_confidence, box_class_probs)
    box_classes = K.argmax(box_scores, axis = -1)
    box_class_scores = K.max(box_scores, axis = -1)

    filtering_mask = K.greater_equal(box_class_scores, threshold)

    boxes = tf.boolean_mask(boxes , filtering_mask)
    scores = tf.boolean_mask(box_class_scores , filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)


    return boxes, classes, scores

with tf.Session() as sess:
    box_confidence = tf.random_normal([19, 19, 5, 1], mean = 1, stddev = 4, seed = 1)
    boxes = tf.random_normal([19, 19, 5, 4], mean = 1, stddev = 4, seed = 1)
    box_class_probs = tf.random_normal([19, 19, 5, 80], mean = 1, stddev = 4, seed = 1)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.6)



#Non-Mxa Suppression.
def iou(box1, box):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box2[2], box2[2])
    yi2 = min(box3[3], box3[3])

    inter_area = (xi2 - xi1) * (yi2 - yi1)

    box1area = (box1[3] - box1[1]) * (box2[2] - box2[0])
    box2area = (box2[3] - box2[1]) * (box2[2] - box2[0])

    union_area = (box1area + box2area) - inter_area
    iou = inter_area / union_area

    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.6):
    """
    Applies Non-max suppression (NMS) set of boxes.
    :param scores:
    :param boxes:
    :param classes:
    :param max_boxes:
    :param iou_threshold:
    :return:
    """

    max_boxes_tensor = K.variable(max_boxes, dtype = 'int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold = iou_threshold)

    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes,nms_indices)
    classes = K.gather(classes, nms_indices)

    return classes, boxes, scores


with tf.Session() as sess:
    scores = tf.random_normal([54, ], mean = 1, stddev = 4, seed = 1)
    boxes = tf.random_normal([54, 4], mean = 1, stddev = 4, seed = 1)
    classes = tf.random_normal([54, ],mean = 1, stddev = 4, seed = 1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.6)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """

    # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=max_boxes,
                                                      iou_threshold=iou_threshold)

    return scores, boxes, classes

with tf.Session() as sess:

    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))


sess = K.get_session()

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)

yolo_model = load_model("model_data/yolo.h5")

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict = {yolo_model.input : image_data, K.learning_phase() : 0


    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join("out", image_file), quality=90)
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes

out_scores, out_boxes, out_classes = predict(sess, "test.jpg")
                                                                                         







































