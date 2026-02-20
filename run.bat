@echo off
echo Starting FastAPI Backend...
start "FastAPI" /B uv run uvicorn app.main:app --reload --port 8000

echo Starting Streamlit Frontend...
start "Streamlit" /B uv run streamlit run streamlit_app.py --server.port 8501

echo Both servers are starting!
echo Backend:  http://localhost:8000/docs
echo Frontend: http://localhost:8501
