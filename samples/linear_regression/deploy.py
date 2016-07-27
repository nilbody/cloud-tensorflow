#!/bin/bash

import argparse
import logging
import os
import subprocess
import uuid

MODEL_NAME = 'mnist'

parser = argparse.ArgumentParser(
    description='Deploys the MNIST model to the Cloud.')
parser.add_argument('--project_id',
                    help='The project to which the job will be submitted.')
parser.add_argument('--version',
                    help='The version of the model to deploy. If omitted, '
                    '3 random characters are appended to a \'v\'.')
parser.add_argument('--source',
                    help='The directory in which the model resides.')
args = parser.parse_args()

version = args.version or ('v' + uuid.uuid4().hex[:2])

model = {'name': MODEL_NAME}
request = {'name': version, 'origin_uri': args.source}

print(
    "Deploy the model and run inference service on the cloud, you can run `ps aux |grep run_deploy` to check service"
)
run_local_service_cmd = ['/tmp/cloud-tensorflow/local_platform/run_deploy.py',
                         MODEL_NAME, version, args.source]
run_deploy_job = subprocess.Popen(run_local_service_cmd, )
#run_deploy_job.wait()

print 'Deployed {}.{}'.format(MODEL_NAME, version)
