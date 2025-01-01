FROM python:3.9.13

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model (if necessary)
# COPY model.pkl /app/model.pkl

# Expose the port the app runs on
EXPOSE 5000


# Run the application
CMD ["python", "/app/src/main/app.py"]