FROM python:3.9-slim


# Install system dependencies for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	gcc \
	libffi-dev \
	libssl-dev \
	libxml2-dev \
	libxslt1-dev \
	libjpeg-dev \
	zlib1g-dev \
	libpng-dev \
	git \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt && rm -rf /root/.cache/pip

# Copy the rest of the application code
COPY . .

# Expose the port that Flask will run on (match Jenkins: 5000)
EXPOSE 5000

# Set the command to run the Flask app
CMD ["python", "app.py", "--host=0.0.0.0", "--port=5000"]