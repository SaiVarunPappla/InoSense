FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENV STREAMLIT_SERVER_ROOT=/tmp \
    STREAMLIT_CONFIG_DIR=/tmp/.streamlit

RUN mkdir -p ${STREAMLIT_CONFIG_DIR} && \
    chmod -R 777 ${STREAMLIT_CONFIG_DIR}

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]