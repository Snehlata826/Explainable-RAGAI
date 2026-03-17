"""
api/client.py
HTTP client for the FastAPI RAG backend.
"""

import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

API_BASE = "http://localhost:8000"


def upload_file(file) -> dict:
    """
    Upload a PDF or TXT file to the /upload endpoint.

    Args:
        file: Streamlit UploadedFile object.

    Returns:
        dict with keys:
          - success (bool)
          - message (str)
          - data (dict | None)  raw JSON from backend on success
    """
    try:
        response = requests.post(
            f"{API_BASE}/upload",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=60,
        )
        if response.status_code == 200:
            return {"success": True, "message": f"{file.name} indexed successfully.", "data": response.json()}
        else:
            return {
                "success": False,
                "message": f"Backend returned {response.status_code}: {response.text[:120]}",
                "data": None,
            }
    except ConnectionError:
        return {"success": False, "message": "Cannot connect to backend (localhost:8000). Is the server running?", "data": None}
    except Timeout:
        return {"success": False, "message": "Upload timed out. Try a smaller file or check backend.", "data": None}
    except RequestException as e:
        return {"success": False, "message": f"Upload error: {str(e)}", "data": None}


def query(question: str) -> dict:
    """
    Send a question to the /query endpoint.

    Args:
        question: User's natural-language question string.

    Returns:
        dict with keys:
          - success (bool)
          - answer (str)   final answer text, or error message
    """
    if not question or not question.strip():
        return {"success": False, "answer": "Please enter a question."}

    try:
        response = requests.post(
            f"{API_BASE}/query",
            json={"question": question.strip()},
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            # Support common response shapes
            answer = (
                data.get("answer")
                or data.get("response")
                or data.get("result")
                or data.get("output")
                or str(data)
            )
            if not answer or not str(answer).strip():
                answer = "No answer returned. Please check your documents and try again."
            return {"success": True, "answer": str(answer)}
        else:
            return {
                "success": False,
                "answer": f"Backend error {response.status_code}: {response.text[:200]}",
            }
    except ConnectionError:
        return {"success": False, "answer": "Cannot connect to backend (localhost:8000). Is the server running?"}
    except Timeout:
        return {"success": False, "answer": "Request timed out. The backend may be overloaded."}
    except RequestException as e:
        return {"success": False, "answer": f"Network error: {str(e)}"}