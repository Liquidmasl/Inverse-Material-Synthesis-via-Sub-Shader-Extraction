FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ADD . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "loadvgg.py"]