# start the container
FROM python:3.11-slim
WORKDIR /app

# copy requirements.txt inside the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all project files into the container
COPY . .

# create an unprivileged user for safety reasons
RUN useradd -m -u 1000 user
USER user

# Container listens on port 7860, the port for HuggingFace Spaces
EXPOSE 7860

# start the application when the container runs
CMD ["python", "main.py"]