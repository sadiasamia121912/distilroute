# Student-only routing service (roadmap 3.4): one fine-tuned int8 ONNX student behind FastAPI,
# no torch and no API key.
#
#   docker build -t distilroute .                                     # minilm_ft_teacher
#   docker build -t distilroute --build-arg MODEL=tinybert_ft_gold .
#   docker run --rm -p 8000:8000 distilroute
#   curl -X POST 127.0.0.1:8000/route -H 'content-type: application/json' \
#        -d '{"text": "my card still has not arrived"}'
#
# models/ is gitignored. The default model, the served student (trained on teacher labels
# only), is a GitHub release asset; unzip it at the repo root first:
#   curl -fsSLO https://github.com/sadiasamia121912/distilroute/releases/download/model-v1/minilm_ft_teacher.zip
#   unzip minilm_ft_teacher.zip        # -> models/minilm_ft_teacher/
# Other students come from notebooks/finetune_colab.ipynb. Only `onnx` ones (*_ft_*) load here.
# CI builds and calls this image on every change: .github/workflows/docker.yml.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

ARG MODEL=minilm_ft_teacher
COPY distilroute/ distilroute/
COPY models/${MODEL}/ models/${MODEL}/
ENV DISTILROUTE_MODEL=${MODEL}

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "distilroute.serve:app", "--host", "0.0.0.0", "--port", "8000"]
