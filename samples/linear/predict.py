import argparse
import datetime
import glob
import itertools
import os
import subprocess
import sys
import tempfile
import json

import requests

MODEL_NAME = 'mnist'


def main():
    parser = argparse.ArgumentParser('Predict on the MNIST model.')

    parser.add_argument('--cloud',
                        action='store_true',
                        help='Prediction should run in the cloud')

    parser.add_argument('input', help='The input data file.')

    parser.add_argument(
        '--version',
        help=('The name of the model version against which to issue the '
              'online prediction requests or run batch prediction. This '
              'would be used to find the actual model location. However, '
              'the location found through model name and model version can '
              'be overridden by --model_dir flag.'))

    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help=('The path to the model where the tensorflow meta graph '
              'proto and checkpoint files are saved. Normally, it is '
              'the exported directory by session_bundle library.'
              'If specified, it overrides the model dir found through '
              'model name and model version.'))

    args = parser.parse_args()

    instances = []
    with open(args.input) as f:
        # We use at most the first 100 entries in each file for prediction.
        for line in itertools.islice(f, 100):
            instances.append(json.loads(line.strip()))

    print(instances)

    protocol = "http"
    host = "127.0.0.1:5000"
    path = '%s://%s' % (protocol, host)

    request_body = json.dumps(instances)
    response = requests.post(path, data=request_body)

    if response.status_code == 200:
        print("Successfully requesti, response data: {}".format(
            response.content))
    else:
        print("Fail to request, status code: {}".format(response.status_code))


if __name__ == '__main__':
    main()
