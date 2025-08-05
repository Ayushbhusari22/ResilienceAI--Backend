# Base Image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create 'model_files' directory inside container (just in case)
RUN mkdir -p model_files

# Expose port 5000 (Flask default)
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]
