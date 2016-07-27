FROM tensorflow/tensorflow
MAINTAINER tobe tobeg3oogle@gmail.com

RUN pip install flask

Add ./model  /tmp/mnist_160727_185504/model/
ADD cmd.py /

CMD /cmd.py
