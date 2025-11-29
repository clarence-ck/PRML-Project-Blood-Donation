FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

# Install build tooling required for scientific Python sdists (numpy/pandas, etc.)
RUN yum install -y gcc gcc-c++ gcc-gfortran make openblas-devel \
    && yum clean all

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models/best_model.pkl ./models/best_model.pkl
COPY app_lambda.py ./app_lambda.py

CMD ["app_lambda.handler"]
