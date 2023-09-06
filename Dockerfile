FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV FLASK_RUN_HOST=0.0.0.0

CMD ["gunicorn", "app:app", "-b", "0.0.0.0:80", "--log-level", "info"]
