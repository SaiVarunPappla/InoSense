FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN mkdir -p /tmp/.streamlit
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]