# backend/Dockerfile
FROM ollama/ollama:latest as builder

ENV OLLAMA_HOST=0.0.0.0:11434

RUN apt-get update && apt-get install -y curl

RUN mkdir -p /root/.ollama
RUN nohup ollama serve > /var/log/ollama.log 2>&1 & \
    sleep 10 && \
    ollama pull command-r7b-arabic && \
    ollama pull granite3.2:2b && \
    ollama pull granite-embedding:278m && \
    sleep 60

FROM ollama/ollama:latest

ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_ORIGINS=*

COPY --from=builder /root/.ollama /root/.ollama

EXPOSE 11434
VOLUME ["/local/repository:/root/.ollama"]
VOLUME /data

ENTRYPOINT ["ollama"]
CMD ["serve"]