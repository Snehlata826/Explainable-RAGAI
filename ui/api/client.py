"""
api/client.py
HTTP client for the FastAPI RAG backend.
"""

import os
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

API_BASE = os.environ.get("API_BASE", "http://localhost:8005")


# ─────────────────────────────────────────────────────────────
# 🔥 RESET VECTOR DATABASE
# ─────────────────────────────────────────────────────────────
def reset_index() -> dict:
    """
    Clear the vector database via /reset endpoint.

    Returns:
        dict with:
          - success (bool)
          - message (str)
    """
    try:
        response = requests.delete(
            f"{API_BASE}/reset",
            timeout=30,
        )

        if response.status_code == 200:
            return {
                "success": True,
                "message": "Vector database cleared successfully."
            }
        else:
            return {
                "success": False,
                "message": f"Reset failed ({response.status_code}): {response.text[:120]}"
            }

    except ConnectionError:
        return {
            "success": False,
            "message": "Cannot connect to backend (localhost:8005). Is server running?"
        }
    except Timeout:
        return {
            "success": False,
            "message": "Reset request timed out."
        }
    except RequestException as e:
        return {
            "success": False,
            "message": f"Reset error: {str(e)}"
        }


# ─────────────────────────────────────────────────────────────
# 📄 UPLOAD FILE
# ─────────────────────────────────────────────────────────────
def upload_file(file) -> dict:
    """
    Upload a PDF or TXT file to the /upload endpoint.

    Args:
        file: Streamlit UploadedFile object.

    Returns:
        dict with:
          - success (bool)
          - message (str)
          - data (dict | None)
    """
    try:
        response = requests.post(
            f"{API_BASE}/upload",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=60,
        )

        if response.status_code == 200:
            return {
                "success": True,
                "message": f"{file.name} indexed successfully.",
                "data": response.json(),
            }
        else:
            return {
                "success": False,
                "message": f"Upload failed ({response.status_code}): {response.text[:120]}",
                "data": None,
            }

    except ConnectionError:
        return {
            "success": False,
            "message": "Cannot connect to backend (localhost:8005). Is server running?",
            "data": None,
        }
    except Timeout:
        return {
            "success": False,
            "message": "Upload timed out. Try a smaller file.",
            "data": None,
        }
    except RequestException as e:
        return {
            "success": False,
            "message": f"Upload error: {str(e)}",
            "data": None,
        }


# ─────────────────────────────────────────────────────────────
# ❓ QUERY
# ─────────────────────────────────────────────────────────────
def query(question: str) -> dict:
    """
    Send a question to the /query endpoint.

    Args:
        question: User question string.

    Returns:
        dict with:
          - success (bool)
          - answer (str)
    """
    if not question or not question.strip():
        return {
            "success": False,
            "answer": "Please enter a valid question."
        }

    try:
        response = requests.post(
            f"{API_BASE}/query",
            json={"question": question.strip()},
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()

            # Flexible response handling
            answer = (
                data.get("answer")
                or data.get("response")
                or data.get("result")
                or data.get("output")
                or str(data)
            )

            if not answer or not str(answer).strip():
                answer = "No answer returned. Please check your documents."

            return {
                "success": True,
                "answer": str(answer)
            }

        else:
            return {
                "success": False,
                "answer": f"Backend error {response.status_code}: {response.text[:200]}"
            }

    except ConnectionError:
        return {
            "success": False,
            "answer": "Cannot connect to backend (localhost:8005). Is server running?"
        }
    except Timeout:
        return {
            "success": False,
            "answer": "Request timed out. Backend may be overloaded."
        }
    except RequestException as e:
        return {
            "success": False,
            "answer": f"Network error: {str(e)}"
        }