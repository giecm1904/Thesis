FROM python:3.8-slim

RUN apt update
RUN apt install nano -y
RUN apt install vim -y
RUN apt install screen -y
RUN apt install curl -y
RUN apt install python3-pyproj -y
RUN apt install git -y
COPY requirements.txt .

RUN pip install -r requirements.txt

RUN git clone https://github.com/sybrenstuvel/Python-RVO2.git
WORKDIR /Python-RVO2
RUN python setup.py build
RUN python setup.py install

WORKDIR ..

COPY *.py ./
COPY *.sh ./
RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["/bin/bash", "./entrypoint.sh"]