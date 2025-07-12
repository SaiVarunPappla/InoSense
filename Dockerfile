# Use official Python 3.9 image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port (change 8050 if your app uses different port)
EXPOSE 8050

# Run the application
CMD ["python", "src/main.py"]