# stilos
FROM python:3.11.4
WORKDIR /app

COPY ./*css /app

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

EXPOSE 8082

ENTRYPOINT [ "streamlit","run","app.py","--server.port=8082","--server.address=0.0.0.0"]