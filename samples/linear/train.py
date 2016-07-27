import argparse
import datetime
import json
import logging
import os
import subprocess
import tarfile
import time

import setup

MODEL_NAME = 'mnist'
JOB_NAME_PREFIX = MODEL_NAME + '_'
PACKAGE = setup.NAME
MODULE = 'task'
TAR_FILE = '{package}-{version}.tar.gz'.format(package=PACKAGE,
                                               version=setup.VERSION)
DEFAULT_MODULE = 'task'

def main():
  parser = argparse.ArgumentParser('Trains the MNIST model.')
  parser.add_argument('--cloud',
                      action='store_true',
                      help='Training should run in the cloud')
  args, passthrough_args = parser.parse_known_args() 
  
  job_name = JOB_NAME_PREFIX + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
  job = {
      'job_name': job_name,
      'module_name': PACKAGE + '.' + MODULE,
      'train_data_paths': "",
      'eval_data_paths': "",
      'job_args': passthrough_args
  }

  if args.cloud:
    #job['output_path'] = _get_output_path(args, 'gs://%s-ml/mnist' %
    #                                      (args.project,), job_name)
    job['output_path'] = os.path.join("/tmp/", job_name) 
    mkdir_cmd = ['mkdir', job['output_path']]
    mkdir_job = subprocess.Popen(mkdir_cmd,)
    mkdir_job.wait()
    
    train_on_cloud(job)
  else:
    #job['output_path'] = _get_output_path(args, 'output', job_name)
    train_locally(job)

def train_on_cloud(job):
  print("Start to train on the cloud")
  
  tar_src = os.path.join('dist', TAR_FILE)
  tar_dest = os.path.join(job['output_path'], TAR_FILE)
  sdist = ['python', 'setup.py', 'sdist', '--format=gztar']
  #copy = ['gsutil', 'cp', tar_src, tar_dest]
  copy = ['cp', tar_src, tar_dest]
  with open('/dev/null', 'w') as dev_null:
    subprocess.check_call(sdist, stdout=dev_null, stderr=dev_null)
  subprocess.check_call(copy)

  # Request API to submmit job
  print(json.dumps(job))
  run_local_service_cmd = ['/tmp/cloud-tensorflow/local_service/run_train.sh', job['output_path'], TAR_FILE, json.dumps(job, indent=2)]
  mkdir_job = subprocess.Popen(run_local_service_cmd,)
  mkdir_job.wait()

def train_locally(job):
  print("Start to train locally")
  cmd = ['python',
         '-m',
         job['module_name'],
         '--job=' + json.dumps(job),]

  job = subprocess.Popen(cmd,)
  job.wait()

if __name__ == '__main__':
  main()
