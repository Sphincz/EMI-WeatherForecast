FROM apache/airflow:3.1.7
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         vim \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
# Copy requirements and install them in the Airflow container
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
# Set the safe directory for git to avoid issues with Airflow's DVC integration
RUN git config --global --add safe.directory '*'