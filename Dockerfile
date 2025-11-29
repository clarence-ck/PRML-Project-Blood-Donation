FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models/best_model.pkl ./models/best_model.pkl
COPY app_lambda.py ./app_lambda.py

CMD ["app_lambda.handler"]
