#!/usr/bin/env python

import tensorflow as tf
import json
import sys
import os

from flask import Flask
from flask import request

app = Flask(__name__)

class InferenceService(object):

    def __init__(self, model, version, checkpoint_dir, graph_dir):
        self.model = model
        self.version = version
        #self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir = "./model/"
        #self.checkpoint_dir = "/home/tobe/code/model-player/model/"
        #self.meta_graph_file = graph_dir
        self.meta_graph_file = "./model/linear_model.ckpt-100.meta"
        #self.meta_graph_file = "/home/tobe/code/model-player/model/"

        self.sess = None
        self.inputs = None
        self.outputs = None
        self.load_model()

    def __del__(self):
        self.sess.close()

    def load_model(self):
        self.sess =  tf.Session()

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.import_meta_graph(self.meta_graph_file)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.inputs = json.loads(tf.get_collection('inputs')[0])
            self.outputs = json.loads(tf.get_collection('outputs')[0])
        else:
            print("No model found, exit")

    def process_json_file(self, json_file):
        result = []
        with open(json_file) as f:
            for line in f.readlines():
                # line = {'key': 1, 'X': 10.0, 'Y': 20.0}
                predict_sample = json.loads(line)
                result.append(self.process_request(predict_sample))
        return result       

    def process_request(self, predict_sample):
        # request_data = {'key': 1, 'X': 10.0, 'Y': 20.0}
        # inputs = {'key_placeholder': placeholder1, 'X': placeholder2, 'Y': placeholder3}
        # outputs = {'key': identity_op, 'predict_op1': predict_op1, 'predict_op2': predict_op2}
        print("Request data: {}".format(predict_sample))
        feed_dict = {}
        for key in self.inputs.keys():
            feed_dict[self.inputs[key]] = predict_sample[key]
        # TODO: this dict not works in docker container
        #response = self.sess.run(self.outputs, feed_dict=feed_dict)
        response = self.sess.run(self.outputs.values(), feed_dict=feed_dict)
        print("Response data: {}".format(response))
        return response
        

service = None

@app.route("/", methods=["GET"])
def index():
    return "Test endpoint"

@app.route("/", methods=["POST"])
def main():
    # Predict request
    #json_file = "./predict_sample.tensor.json"
    #response =  service.process_json_file(json_file)

    data = json.loads(request.data)
  
    result = [] 
    for predict_sample in data:
        response = service.process_request(predict_sample)
        result.append(response)

    return str(result)



if __name__ == "__main__":

    model = sys.argv[1]
    version = sys.argv[2]
    source= sys.argv[3]

    # TODO: move checkpoint to somewhere
    checkpoint_dir = source
    graph_dir = os.path.join(source, "graph.pb")

    global service
    service = InferenceService(model, version, checkpoint_dir, graph_dir)

    app.run(host="0.0.0.0")

