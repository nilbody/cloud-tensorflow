## Introduction

[Cloud TensorFlow](https://github.com/tobegit3hub/cloud-tensorflow) is the scalable service to run [TensorFlow](https://github.com/tensorflow/tensorflow) in [Kubernetes](https://github.com/kubernetes/kubernetes) cloud platform.

Now you can train TensorFlow models in container cluster with GPUs and deploy inference service with the [Google Cloud ML](https://cloud.google.com/ml/)-like APIs.

## Deployment

For quick start, highly recommend to use local platform which is easy to deploy.

The Kubernetes platform is under development. It is highly relied on storage and you can extend for your requirements.

### Local Platform

The scripts to train model and setup inference service are in [local_platform](./local_platform/). All you need is placing them in `/tmp`.

```
git clone https://github.com/tobegit3hub/cloud-tensorflow.git

mv ./cloud-tensoflow/ /tmp/
```

### Kubernetes Platform

You need to setup [minikube](https://github.com/kubernetes/minikube) or Kubernetes cluster at first.

## Usage

### Prepare Data

As the way to use TensorFlow, we can write Python script to train model. In order to run training in the cloud platform, we need extra scripts to submit training and deploying jobs.

The basic struct of code looks like this and you can find the complete example in [linear_regression](./samples/linear_regression/).

```
├── data
│   └── predict_sample.tensor.json
├── deploy.py
├── predict.py
├── setup.py
├── trainer
│   ├── __init__.py
│   └── task.py
└── train.py
```

### Train Model

The core of your model is written in `trainer/task.py` and you can run that moduel with `train.py`.

```
python ./train.py
```

With the parameter `--cloud`, the script will package the module and upload to train in cloud platform with infrustrature resources.

```
python ./train.py --cloud
```

### Deploy Model

Once you generate the checkpoint files, you can specify the source and deploy as an inference service in cloud platform.

```
python deploy.py --version v1 --source /tmp/mnist_160727_154539/model
```

### Predict Service

Now you submit the predict job or access your online service with the script and data.

```
python predict.py --version v1 data/predict_sample.tensor.json
```
