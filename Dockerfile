# Python 3.11 slim image as base
FROM python:3.11-slim

# Working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Installing dependencies without hash verification to avoid problems with the slim image
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --no-deps --ignore-installed -r requirements.txt \
    --index-url https://pypi.org/simple \
    --timeout 100


# Copy the rest of the application
COPY . .

# Port exposed
EXPOSE 8000

# Command to run the FastAPI application with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]