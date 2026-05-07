# SAM 3 video-tracking handler — small custom image so we can talk to
# the pod over plain HTTP and skip the SSH key/auth dance.
#
# Built and pushed by the GitHub Actions workflow at
# .github/workflows/sam3-video-pod.yml; the pod just pulls the tagged
# image. The runpod/pytorch:2.4.0 base already has torch + cuda; we
# add transformers, FastAPI, our handler, and pre-cache the SAM 3
# weights so first-call latency is minimal.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache

RUN pip install --no-cache-dir \
        "transformers==5.7.0" \
        "accelerate>=0.34" \
        "safetensors" \
        "pillow" \
        "numpy" \
        "opencv-python-headless" \
        "imageio[ffmpeg]" \
        "fastapi" \
        "uvicorn[standard]" \
        "python-multipart"

# Note: NOT baking SAM 3 weights into the image. Weights are ~3 GB
# and would inflate image size + build time. The handler downloads
# them on the first POST (~60–90 s on a fresh pod) and HF caches
# them in /opt/hf-cache for the rest of the pod's life. Trade: first
# call slower, every other call same; image stays ~1.5 GB.

WORKDIR /app
COPY handler.py /app/handler.py
COPY chunking.py /app/chunking.py

EXPOSE 9001
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "9001", "--workers", "1"]
