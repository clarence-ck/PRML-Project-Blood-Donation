FROM public.ecr.aws/lambda/python:3.11-al2023

WORKDIR /var/task

# Install build tooling required for compiling scientific Python sdists if wheels are unavailable
RUN dnf install -y gcc gcc-c++ gcc-gfortran make \
    && dnf clean all

ENV CC=gcc CXX=g++ FC=gfortran

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models/best_model.pkl ./models/best_model.pkl
COPY app_lambda.py ./app_lambda.py

CMD ["app_lambda.handler"]
