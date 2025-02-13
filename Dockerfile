FROM python:3.10-slim

RUN pip3 install --upgrade pip
RUN apt-get update

WORKDIR /app
COPY . /app

RUN pip3 install -r /app/requirements.txt

COPY index.html /usr/local/lib/python3.10/site-packages/streamlit/static/index.html

RUN cd /app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=5006", "--server.address=0.0.0.0"]
