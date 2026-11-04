# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and model artifact into container
COPY app.py /app/app.py
COPY model.pkl /app/model.pkl

# Expose port 5000 for Flask API access
EXPOSE 5000

# Run Flask backend server using Gunicorn WSGI container
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
