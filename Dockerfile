FROM python:3.8-slim-buster

#WORKDIR ./fuseapp/point-e
RUN apt-get update && apt-get install -y git
#RUN pip3 install -e .
RUN git clone https://github.com/openai/point-e

WORKDIR /point-e

RUN pip3 install -e .

RUN python3 setup.py egg_info
WORKDIR /point-e/point_e.egg-info
RUN pip3 install -r requires.txt
WORKDIR /
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
WORKDIR /pages
CMD [ "python3", "app.py"]
