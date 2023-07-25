FROM python:3.10.11
MAINTAINER Tyller
WORKDIR /usr/src/app
COPY . .


RUN pip install -r requirements.txt

CMD ["python", "flask/run.py"]