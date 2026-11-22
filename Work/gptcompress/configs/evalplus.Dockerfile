FROM python:3.12-slim
RUN python -m pip install --no-cache-dir evalplus==0.3.1
ENV PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/cache
USER 65534:65534
ENTRYPOINT ["python", "-m", "evalplus.evaluate"]
