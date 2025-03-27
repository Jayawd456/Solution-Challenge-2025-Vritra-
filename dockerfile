# Use a lightweight Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first (better Docker caching)
COPY requirements.txt .

# Ensure requirements.txt exists before installing
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy the rest of the project files
COPY . .

# Expose the port Flask runs on (Modify if using a different port)
EXPOSE 7860

# Set environment variables to prevent Python buffering issues
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production  
# Change to development if debugging

# Run the Flask app with Gunicorn for better performance (2 workers)
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:7860", "app:app"]
