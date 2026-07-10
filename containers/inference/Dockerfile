ARG BASE_IMAGE=body-engine-base:local
FROM ${BASE_IMAGE}
COPY app app
COPY training training
COPY agents agents
COPY inference inference
COPY containers/runtime_entrypoint.py /usr/local/bin/body-engine-entrypoint
RUN chmod 0555 /usr/local/bin/body-engine-entrypoint && chown -R bodyengine:bodyengine /opt/body-engine
USER bodyengine
EXPOSE 8080
ENTRYPOINT ["python", "/usr/local/bin/body-engine-entrypoint", "inference"]
