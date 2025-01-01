FROM python:3.9.13

# Set the working directory in the container
WORKDIR /app

# Set the Python path to include /app
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV DATA_FILE_PATH=/app/Twitter_Sentiment_Indian_Election_2019/data/Twitter_Data.csv


# Copy only the necessary files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5001

# Run the application
CMD ["python", "/app/Twitter_Sentiment_Indian_Election_2019/src/main/app.py"]
