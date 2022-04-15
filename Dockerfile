FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

COPY kaigeo kaigeo

ENV PYTHONUNBUFFERED=1

RUN pip install -e kaigeo

CMD geometry-postprocess
