FROM python:3.10-slim

RUN pip3 install --upgrade pip
RUN apt-get update && apt-get install -y unzip

WORKDIR /app

COPY requirements.txt /app
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -r /app/requirements.txt

COPY . /app

COPY index.html /usr/local/lib/python3.10/site-packages/streamlit/static/index.html

RUN cd /app

RUN mkdir -p /root/.cache/huggingface/hub
RUN unzip /app/model/models--BAAI--bge-small-zh-v1.5.zip -d /root/.cache/huggingface/hub
RUN python3 encoder.py  # check embedding model works well
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=5006", "--server.address=0.0.0.0"]
