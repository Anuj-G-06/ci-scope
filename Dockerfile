# shrapnel: pick the sqlglot tests a patch can hit.
# Pinned Python 3.13 base image, so the build is the same every time.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# Only the TPC-DS check subprocess imports these, at the versions sqlglot's tests were labeled with.
RUN pip install --no-cache-dir --no-deps numpy==2.5.5 pandas==3.1.1 python-dateutil==4.0.0 tzdata==2026.6 pytz==2026.6

COPY app /app
COPY data /app/data
# Precompute every source file's structure once, so the selector never parses whole files.
RUN python -I -S -B /app/build_structure.py /app/data && chmod -R a+rX /app

CMD ["python", "-I", "-S", "-B", "/app/main.py"]
