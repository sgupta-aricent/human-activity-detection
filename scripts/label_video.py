# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import argparse
import sys
import time
import cv2
import math
import numpy as np
import tensorflow as tf

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
writer = None


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(
        dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == "__main__":
    file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
    model_file = "tf_files/retrained_graph.pb"
    label_file = "tf_files/retrained_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 128
    input_std = 128
    input_layer = "Mul"
    output_layer = "final_result"
    min_accuracy = 60

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.video:
        file_name = args.video
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    graph = load_graph(model_file)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    labels = load_labels(label_file)
    statsArray = [0]*len(labels)
    min_accuracy = min_accuracy/100
    if os.path.exists("screens"):
        shutil.rmtree("screens")
    if os.path.exists("recognized.avi"):
        os.remove("recognized.avi")
    
    i = 0

    with tf.Session(graph=graph) as sess:
        video_capture = cv2.VideoCapture(file_name)
        pathOut = "screens"
        os.mkdir(pathOut)
        while (video_capture.isOpened()):
            print("Video is opened")
            ret, frame = video_capture.read()
            # write frame image to file
            if ret == True:
                i = i + 1
                cv2.imwrite(os.path.join(
                            pathOut, "frame{:d}.jpg".format(i)),
					        frame)
                t = read_tensor_from_image_file(os.path.join(
                                                pathOut, "frame{:d}.jpg".format(i)),
					                            input_height=input_height,
					                            input_width=input_width,
					                            input_mean=input_mean,
					                            input_std=input_std)
                results = sess.run(output_operation.outputs[0],
				                   {input_operation.outputs[0]: t})
				# analyse the image
                top_k = results[0].argsort()[-len(results[0]):][::-1]
                pos = 1
                print("Read %d frame" % i, ret)
                for node_id in top_k:
                    human_string = labels[node_id]
                    score = results[0][node_id]
                    if(score >= min_accuracy):
                        statsArray[node_id] = statsArray[node_id] + 1
                    cv2.putText(frame, '%s (score = %.5f)' % (human_string, score),
							(40, 40 * pos), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255))
                    print('%s (score = %.5f)' % (human_string, score))
                    pos = pos + 1
                print("\n\n")
                if writer is None:
				# initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter("recognized.avi", fourcc, 10,
                                     (frame.shape[1], frame.shape[0]), True)

			# write the output frame to disk
                writer.write(frame)
                cv2.imshow("image", frame)  # show frame in window
                cv2.waitKey(1)  # wait 1ms -> 0 until key input
            else:
                print("Unable to read frame")
                break
        if writer is not None:
	        writer.release()
        video_capture.release()
        cv2.destroyAllWindows()
        if i>0:
            print("Overall statistics of action performed with certainity of %.2f" % ((min_accuracy)*100))
            predictedActions = 0;
            for index in range(len(labels)):
                statValue = (statsArray[index]/i)
                predictedActions = predictedActions + statValue
                print("%s : %.2f" % (labels[index], (statValue*100)))		
            print("Unrecognised action %.2f" % ((1-predictedActions)*100)) 