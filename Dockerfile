# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run Streamlit or Gradio when the container launches
# For Streamlit
ENTRYPOINT ["streamlit", "run"]
CMD ["streamlit_app.py"]

# For Gradio (comment out the Streamlit lines and uncomment these lines)
# ENTRYPOINT ["python"]
# CMD ["gradio_app.py"]
