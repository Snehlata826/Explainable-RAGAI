import requests

API_BASE = "http://localhost:8000"


def upload_file(file):

    r = requests.post(
        f"{API_BASE}/upload",
        files={"file": (file.name, file.getvalue(), file.type)},
    )

    return r.json()


def query(question):

    r = requests.post(
        f"{API_BASE}/query",
        json={"question": question},
    )

    return r.json()