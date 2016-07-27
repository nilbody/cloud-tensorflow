
python ./train.py

python ./train.py --cloud


python deploy.py --version v7 --source /tmp/mnist_160727_154539/model

python predict.py --version v1 data/predict_sample.tensor.json
