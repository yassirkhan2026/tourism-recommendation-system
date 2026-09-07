
# Use Python 3.13 as the base image
FROM python:3.13

# Set the working directory
WORKDIR /app

# Copy deployment files into the container
COPY . .

# Install the required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the application working directory
WORKDIR $HOME/app

# Copy files with appropriate ownership
COPY --chown=user . $HOME/app

# Expose Streamlit's default port
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
