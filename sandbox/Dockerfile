FROM python:3.11-slim

# No network access at runtime — enforced by docker run --network none
# Only the libraries the code agent is allowed to use
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    matplotlib==3.9.0 \
    scipy==1.13.1 \
    numpy==1.26.4 \
    openpyxl==3.1.2

# Drop root — run as nobody
RUN useradd -m sandbox
USER sandbox

WORKDIR /home/sandbox
COPY runner.py .

ENTRYPOINT ["python", "runner.py"]