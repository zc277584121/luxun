FROM python:3.10-slim

RUN pip3 install --upgrade pip
RUN apt-get update

WORKDIR /app/src
COPY . /app

RUN pip3 install -r /app/requirements.txt

RUN cd /app/src
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=5005", "--server.address=0.0.0.0"]
