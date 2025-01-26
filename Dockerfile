# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir flask scikit-learn

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
