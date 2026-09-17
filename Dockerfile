# Use official slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy dataset, models, source code, and application scripts
COPY data/ data/
COPY models/ models/
COPY results/ results/
COPY src/ src/
COPY main.py .
COPY app.py .

# Expose Streamlit default port
EXPOSE 8501

# Default command to run CLI application (can be overridden to streamlit run app.py)
CMD ["python", "main.py"]
