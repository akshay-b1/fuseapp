FROM python:3.8-slim-buster

WORKDIR /
RUN git clone https://github.com/openai/point-e

WORKDIR /point-e


RUN pip install -e .


WORKDIR /
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
