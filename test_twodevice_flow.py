import asyncio
import json
from fastapi.testclient import TestClient
from backend.main import app

def test_api():
    client = TestClient(app)
    
    # 1. Test /api/info
    res_info = client.get("/api/info")
    print("API /api/info response:", res_info.json())
    assert res_info.status_code == 200
    assert "local_ip" in res_info.json()
    
    # 2. Test /api/session/create
    res_create = client.post("/api/session/create", json={"language": "ta-IN"})
    print("API /api/session/create response:", res_create.json())
    assert res_create.status_code == 200
    session_id = res_create.json()["session_id"]
    
    # 3. Test /api/session/{session_id}
    res_get = client.get(f"/api/session/{session_id}")
    print(f"API /api/session/{session_id} response:", res_get.json())
    assert res_get.status_code == 200
    assert res_get.json()["exists"] is True

    print("\nAll REST API endpoints for two-device sessions PASSED!\n")

if __name__ == "__main__":
    test_api()
