import asyncio
import json
from fastapi.testclient import TestClient
from backend.main import app

def test_ws():
    client = TestClient(app)
    
    # Device A creates session
    with client.websocket_connect("/ws/session") as ws_a:
        ws_a.send_json({"action": "create_session", "language": "ta-IN"})
        res_a1 = ws_a.receive_json()
        print("Device A got:", res_a1)
        assert res_a1["type"] == "session.created"
        session_id = res_a1["session_id"]
        
        # Device B joins session
        with client.websocket_connect("/ws/session") as ws_b:
            ws_b.send_json({"action": "join_session", "session_id": session_id, "language": "hi-IN"})
            res_b1 = ws_b.receive_json()
            print("Device B got:", res_b1)
            assert res_b1["type"] == "session.joined"
            assert res_b1["other_language"] == "ta-IN"
            
            # Device A should receive peer.connected
            res_a2 = ws_a.receive_json()
            print("Device A got (peer.connected):", res_a2)
            assert res_a2["type"] == "peer.connected"
            assert res_a2["language"] == "hi-IN"
            
            # Device A changes language to Telugu
            ws_a.send_json({"action": "change_language", "language": "te-IN"})
            res_a3 = ws_a.receive_json()
            print("Device A got (language.updated):", res_a3)
            assert res_a3["type"] == "language.updated"
            
            # Device B should receive peer.language_changed
            res_b2 = ws_b.receive_json()
            print("Device B got (peer.language_changed):", res_b2)
            assert res_b2["type"] == "peer.language_changed"
            assert res_b2["language"] == "te-IN"

    print("\nTwo-Device WebSocket Session flow PASSED successfully!\n")

if __name__ == "__main__":
    test_ws()
