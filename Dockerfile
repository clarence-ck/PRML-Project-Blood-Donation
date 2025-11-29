FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

# Install build tooling (gcc10) required for compiling newer numpy/pandas releases
RUN yum install -y gcc10 gcc10-c++ gcc10-gfortran make \
    && yum clean all

ENV CC=gcc10 CXX=g++10 FC=gfortran10

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models/best_model.pkl ./models/best_model.pkl
COPY app_lambda.py ./app_lambda.py

CMD ["app_lambda.handler"]
