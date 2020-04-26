FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

WORKDIR /tf
ADD /. /tf
RUN pip install -r requirements.txt

