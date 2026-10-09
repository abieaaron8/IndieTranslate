"""
backend/session_manager.py

In-memory Two-Device Session Manager for IndieTranslate.
Supports:
- Creating 6-character session codes
- Managing two participants per session (A & B)
- Tracking per-participant language, WebSocket, and orchestration engines
- Thread-safe / async-safe WebSocket event dispatching
"""

import secrets
import string
import asyncio
import logging
from typing import Dict, Optional, Tuple, List, Any
from fastapi import WebSocket

from stability_analyzer import ASRStabilityAnalyzer
from commit_engine import CommitEngine

logger = logging.getLogger("session-manager")

# Alphabet for readable 6-character session codes (omits ambiguous 0, O, 1, I)
SESSION_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def generate_session_code(length: int = 6) -> str:
    """Generate a readable uppercase alphanumeric session code."""
    return "".join(secrets.choice(SESSION_CODE_ALPHABET) for _ in range(length))


class Participant:
    """
    Encapsulates all independent state for one connected participant.
    Per-participant independent:
    - WebSocket connection & send lock
    - Selected language
    - ASRStabilityAnalyzer instance
    - CommitEngine instance
    - Utterance counter
    - Audio queue & listening state
    """
    def __init__(self, participant_id: str, websocket: WebSocket, language: str):
        self.participant_id = participant_id
        self.websocket = websocket
        self.language = language

        # Independent orchestration engines
        self.stability_analyzer = ASRStabilityAnalyzer()
        self.commit_engine = CommitEngine(language_code=language)
        self.utterance_idx = 0

        # Audio streaming & state
        self.is_listening = False
        self.listening_event = asyncio.Event()
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.send_lock = asyncio.Lock()
        self.asr_task: Optional[asyncio.Task] = None

    def update_language(self, new_language: str):
        """Update participant language and reconfigure commit engine."""
        self.language = new_language
        self.commit_engine = CommitEngine(language_code=new_language)
        self.stability_analyzer.reset()

    async def send_json(self, data: Dict[str, Any]) -> bool:
        """Send a JSON message safely over this participant's WebSocket."""
        async with self.send_lock:
            try:
                await self.websocket.send_json(data)
                return True
            except Exception as e:
                logger.warning(f"Error sending message to {self.participant_id}: {e}")
                return False


class ConversationSession:
    """
    Represents a 2-person bidirectional conversation session.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id.upper()
        self.participant_a: Optional[Participant] = None
        self.participant_b: Optional[Participant] = None

    def add_participant(self, participant_id: str, websocket: WebSocket, language: str) -> Participant:
        """Add participant to session as A or B."""
        if self.participant_a is None:
            p = Participant(participant_id, websocket, language)
            self.participant_a = p
            return p
        elif self.participant_b is None:
            p = Participant(participant_id, websocket, language)
            self.participant_b = p
            return p
        else:
            raise ValueError(f"Session {self.session_id} is full (maximum 2 participants).")

    def get_participant(self, participant_id: str) -> Optional[Participant]:
        if self.participant_a and self.participant_a.participant_id == participant_id:
            return self.participant_a
        if self.participant_b and self.participant_b.participant_id == participant_id:
            return self.participant_b
        return None

    def get_other_participant(self, participant_id: str) -> Optional[Participant]:
        """Get the peer participant for opposite-device routing."""
        if self.participant_a and self.participant_a.participant_id == participant_id:
            return self.participant_b
        if self.participant_b and self.participant_b.participant_id == participant_id:
            return self.participant_a
        return None

    def remove_participant(self, participant_id: str) -> Optional[Participant]:
        removed = None
        if self.participant_a and self.participant_a.participant_id == participant_id:
            removed = self.participant_a
            self.participant_a = None
        elif self.participant_b and self.participant_b.participant_id == participant_id:
            removed = self.participant_b
            self.participant_b = None
        return removed

    def is_empty(self) -> bool:
        return self.participant_a is None and self.participant_b is None

    def is_full(self) -> bool:
        return self.participant_a is not None and self.participant_b is not None

    def participant_count(self) -> int:
        count = 0
        if self.participant_a is not None:
            count += 1
        if self.participant_b is not None:
            count += 1
        return count


class SessionManager:
    """
    Singleton manager tracking all in-memory active conversation sessions.
    """
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self._lock = asyncio.Lock()

    def create_session(self, preferred_code: Optional[str] = None) -> ConversationSession:
        """Create a new unique conversation session."""
        code = preferred_code.upper() if preferred_code else generate_session_code()
        # Ensure code uniqueness
        while code in self.sessions:
            code = generate_session_code()

        session = ConversationSession(code)
        self.sessions[code] = session
        logger.info(f"Created session {code}")
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        return self.sessions.get(session_id.strip().upper())

    def remove_session(self, session_id: str):
        code = session_id.strip().upper()
        if code in self.sessions:
            del self.sessions[code]
            logger.info(f"Removed session {code}")


# Global in-memory session manager instance
session_manager = SessionManager()
