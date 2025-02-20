FROM python:3.10-slim

RUN pip3 install --upgrade pip
RUN apt-get update

WORKDIR /app

COPY requirements.txt /app
RUN pip3 install -r /app/requirements.txt

COPY . /app

COPY index.html /usr/local/lib/python3.10/site-packages/streamlit/static/index.html

RUN unzip /app/model/models--GPTCache--paraphrase-albert-onnx.zip -d /root/.cache/huggingface/hub
RUN unzip /app/model/models--GPTCache--paraphrase-albert-small-v2.zip -d /root/.cache/huggingface/hub

RUN cd /app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=5006", "--server.address=0.0.0.0"]
