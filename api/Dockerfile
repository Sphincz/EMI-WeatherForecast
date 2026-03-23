# api/Dockerfile
FROM python:3.12-slim

WORKDIR /code
COPY ./api/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./api/app /code/app

# Start the FastAPI application using the production-ready CLI
CMD ["fastapi", "dev", "app/main.py", "--port", "80"]