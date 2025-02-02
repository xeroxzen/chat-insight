# Python 3.12 slim image as base
FROM python:3.12-slim

# Working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies without hash verification
RUN pip install --no-cache-dir --no-deps --ignore-installed -r requirements.txt

# Copy the rest of the application
COPY . .

# Port exposed
EXPOSE 8000

# Command to run the FastAPI application with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]