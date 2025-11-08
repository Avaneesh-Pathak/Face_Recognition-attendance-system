#!/bin/bash

echo "Applying database migrations..."
python manage.py migrate

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Django server with Gunicorn..."
gunicorn attendance.wsgi:application --bind 0.0.0.0:8000
