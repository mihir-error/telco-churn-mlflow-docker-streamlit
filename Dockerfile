FROM python:3.11

WORKDIR /app

COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Add local bin to PATH
ENV PATH="/root/.local/bin:$PATH"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "train_ml.py", "--server.port=8501", "--server.address=0.0.0.0"]
