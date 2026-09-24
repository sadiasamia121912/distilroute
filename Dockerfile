# Student-only routing service (roadmap 3.4): one fine-tuned int8 ONNX student behind FastAPI,
# no torch and no API key.
#
#   docker build -t distilroute .                                     # minilm_ft_gold
#   docker build -t distilroute --build-arg MODEL=tinybert_ft_gold .
#   docker run --rm -p 8000:8000 distilroute
#   curl -X POST 127.0.0.1:8000/route -H 'content-type: application/json' \
#        -d '{"text": "my card still has not arrived"}'
#
# models/ is gitignored: build from a checkout where models/<MODEL>/ exists
# (notebooks/finetune_colab.ipynb writes it). Only `onnx` students (*_ft_*) load here.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

ARG MODEL=minilm_ft_gold
COPY distilroute/ distilroute/
COPY models/${MODEL}/ models/${MODEL}/
ENV DISTILROUTE_MODEL=${MODEL}

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "distilroute.serve:app", "--host", "0.0.0.0", "--port", "8000"]
