FROM ubuntu:latest
LABEL authors="science"

ENTRYPOINT ["top", "-b"]