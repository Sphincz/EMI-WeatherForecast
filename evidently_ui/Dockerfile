# evidently_ui/Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY ./evidently_ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Start the Evidently UI server pointing to the mounted workspace
CMD ["evidently", "ui", "--workspace", "/app/workspace", "--host", "0.0.0.0", "--port", "8081"]