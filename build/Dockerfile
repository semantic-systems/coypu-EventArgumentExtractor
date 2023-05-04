FROM python:3.9-slim

RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get -y dist-upgrade && \
    apt-get -y autoremove && \
    apt-get -y install \
        git \
        curl \
    && \
    apt-get -y clean

COPY . .
RUN pip install -r requirements.txt

WORKDIR .

EXPOSE 5287/tcp

ENTRYPOINT ["python"]
CMD ["main.py"]
