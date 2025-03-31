# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables to avoid writing pyc files and buffering
ENV ENVIRONMENT=production
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt /app/
# Copy the .env file to the container
COPY .env /app/.env

# Install the dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire content of your project folder to the working directory
COPY . /app/

# Expose port 8080 to be used by the FastAPI app
EXPOSE 8080

# Run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]