# Reproducible environment for the full pipeline.
#
#   docker build -t coupon-incrementality .
#   docker run --rm -v "$PWD/reports:/app/reports" coupon-incrementality          # make all
#   docker run --rm -p 8501:8501 coupon-incrementality make app                    # simulator
FROM python:3.13-slim

# libgomp: LightGBM's OpenMP runtime. git: required by dbt. make: runs the pipeline.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir -e . --no-deps

COPY . .
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 STREAMLIT_SERVER_HEADLESS=true
EXPOSE 8501
CMD ["make", "all"]
