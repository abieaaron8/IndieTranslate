#!/usr/bin/env python3
"""
verify_project.py

Comprehensive Non-Destructive Verification Suite for IndieTranslate
===================================================================
Performs non-destructive validation of:
  - Hardware & Platform: CUDA / CPU detection, CTranslate2 acceleration
  - Required Folders & Files: backend, frontend, models, configs
  - Environment Variables: .env existence, API keys, backend settings
  - Network & Port Availability: Port 8000 (Backend) & Port 5173 (Frontend)
  - Python Dependencies: Core web, ML, ASR, translation, and tokenization packages
  - Node & Frontend Dependencies: Node.js, npm, package.json, node_modules
  - Faster Whisper Model Integrity: Model weights, config, tokenizer, vocab files
  - IndicTrans2 Model Integrity: Safetensors/bin, architecture, dynamic modules, vocab files
  - Tokenizer Loading: Faster-Whisper and IndicTrans2 tokenizers
  - Model Loading Time: Faster-Whisper and IndicTrans2 loading benchmarks & inference
  - FastAPI Application Startup: App initialization, route registration, CORS
  - OpenAPI Endpoint: /openapi.json validation, /docs swagger check
  - Health Endpoint: /health schema and status validation
  - WebSocket Handshake: Live handshake and session creation over /ws/session

Usage:
  python verify_project.py [OPTIONS]

Options:
  --skip-model-load   Skip loading heavy model weights into memory (fast mode)
  --skip-ws           Skip WebSocket handshake test
  --no-color          Disable ANSI color output
  --no-venv-switch    Do not re-execute inside backend/.venv if detected
  --json              Output verification results as machine-readable JSON
  -v, --verbose       Print verbose tracebacks and diagnostic information
  -h, --help          Show this help message and exit
"""

import sys
import os
import time
import json
import socket
import platform
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Enable ANSI colors and UTF-8 encoding on Windows Console
if os.name == "nt":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend"
FRONTEND_DIR = REPO_ROOT / "frontend"
MODELS_DIR = REPO_ROOT / "models"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ── ANSI Terminal Colors ──────────────────────────────────────────────────────
class Colors:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    @property
    def RESET(self) -> str:
        return "\033[0m" if self.enabled else ""

    @property
    def BOLD(self) -> str:
        return "\033[1m" if self.enabled else ""

    @property
    def DIM(self) -> str:
        return "\033[2m" if self.enabled else ""

    @property
    def RED(self) -> str:
        return "\033[91m" if self.enabled else ""

    @property
    def GREEN(self) -> str:
        return "\033[92m" if self.enabled else ""

    @property
    def YELLOW(self) -> str:
        return "\033[93m" if self.enabled else ""

    @property
    def BLUE(self) -> str:
        return "\033[94m" if self.enabled else ""

    @property
    def MAGENTA(self) -> str:
        return "\033[95m" if self.enabled else ""

    @property
    def CYAN(self) -> str:
        return "\033[96m" if self.enabled else ""

    @property
    def WHITE(self) -> str:
        return "\033[97m" if self.enabled else ""


# ── Check Data Structure ──────────────────────────────────────────────────────
class CheckResult:
    def __init__(
        self,
        name: str,
        category: str,
        status: str,  # "PASS", "FAIL", "WARN", "SKIP", "INFO"
        message: str = "",
        details: str = "",
        remediation: str = "",
        critical: bool = True,
        duration_s: float = 0.0,
    ):
        self.name = name
        self.category = category
        self.status = status
        self.message = message
        self.details = details
        self.remediation = remediation
        self.critical = critical
        self.duration_s = duration_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "remediation": self.remediation,
            "critical": self.critical,
            "duration_s": round(self.duration_s, 4),
        }


# ── Verification Suite ────────────────────────────────────────────────────────
class VerificationSuite:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.colors = Colors(enabled=not args.no_color)
        self.results: List[CheckResult] = []
        self.t_start = 0.0

    def add_result(self, result: CheckResult):
        self.results.append(result)

    def print_section(self, title: str):
        if self.args.json:
            return
        c = self.colors
        print(f"\n{c.BOLD}{c.CYAN}[ CATEGORY: {title.upper()} ]{c.RESET}")

    # ── 1. CUDA & CPU Detection ───────────────────────────────────────────────
    def check_hardware_and_platform(self):
        self.print_section("Hardware & Platform Detection")
        cat = "Hardware & Platform"

        # Platform info
        plat_info = f"{platform.system()} {platform.release()} ({platform.machine()}), {os.cpu_count() or 'unknown'} CPU cores"
        self.add_result(CheckResult(
            name="Operating System & Architecture",
            category=cat,
            status="PASS",
            message=plat_info,
            critical=False,
        ))

        # PyTorch CUDA
        try:
            import torch
            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                dev_count = torch.cuda.device_count()
                dev_name = torch.cuda.get_device_name(0)
                vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                cuda_ver = torch.version.cuda or "N/A"
                self.add_result(CheckResult(
                    name="PyTorch CUDA Acceleration",
                    category=cat,
                    status="PASS",
                    message=f"Enabled ({dev_name}, {vram_gb} GB VRAM, CUDA {cuda_ver}, {dev_count} device(s))",
                    critical=False,
                ))
            else:
                threads = torch.get_num_threads()
                self.add_result(CheckResult(
                    name="PyTorch CUDA Detection",
                    category=cat,
                    status="PASS",
                    message=f"CPU Mode active ({threads} threads, {os.cpu_count()} CPU cores)",
                    details="PyTorch is running on CPU. A CUDA-compatible GPU/driver will significantly accelerate transcription & translation latency.",
                    critical=False,
                ))
        except ImportError:
            self.add_result(CheckResult(
                name="PyTorch CUDA Detection",
                category=cat,
                status="FAIL",
                message="PyTorch is not installed",
                remediation="Run: pip install torch",
                critical=True,
            ))

        # CTranslate2 CUDA
        try:
            import ctranslate2
            ct2_cuda_count = ctranslate2.get_cuda_device_count()
            if ct2_cuda_count > 0:
                self.add_result(CheckResult(
                    name="CTranslate2 Hardware Acceleration",
                    category=cat,
                    status="PASS",
                    message=f"{ct2_cuda_count} CUDA device(s) detected for Faster-Whisper",
                    critical=False,
                ))
            else:
                self.add_result(CheckResult(
                    name="CTranslate2 Hardware Acceleration",
                    category=cat,
                    status="PASS",
                    message="CPU mode active for Faster-Whisper",
                    critical=False,
                ))
        except ImportError:
            self.add_result(CheckResult(
                name="CTranslate2 Acceleration",
                category=cat,
                status="FAIL",
                message="CTranslate2 is not installed",
                remediation="Run: pip install ctranslate2",
                critical=True,
            ))

    # ── 2. Required Folders and Files ─────────────────────────────────────────
    def check_required_folders_and_files(self):
        self.print_section("Directory & File Structure")
        cat = "File Structure"

        required_dirs = [
            ("backend", BACKEND_DIR),
            ("frontend", FRONTEND_DIR),
            ("models", MODELS_DIR),
            ("models/faster-whisper-small", MODELS_DIR / "faster-whisper-small"),
            ("models/indictrans2-indic-indic-320M", MODELS_DIR / "indictrans2-indic-indic-320M"),
            ("frontend/src", FRONTEND_DIR / "src"),
        ]
        missing_dirs = [name for name, p in required_dirs if not p.is_dir()]
        if missing_dirs:
            self.add_result(CheckResult(
                name="Repository Directories",
                category=cat,
                status="FAIL",
                message=f"Missing directories: {', '.join(missing_dirs)}",
                details=f"Required directory tree is incomplete.",
                remediation="Ensure the project repository was cloned completely.",
                critical=True,
            ))
        else:
            self.add_result(CheckResult(
                name="Repository Directories",
                category=cat,
                status="PASS",
                message=f"All {len(required_dirs)} required directories present",
                critical=True,
            ))

        backend_files = [
            "main.py",
            "whisper_asr.py",
            "indictrans2_translator.py",
            "commit_engine.py",
            "completeness_analyzer.py",
            "context_analyzer.py",
            "entity_protector.py",
            "session_manager.py",
            "stability_analyzer.py",
        ]
        missing_backend = [f for f in backend_files if not (BACKEND_DIR / f).is_file() or (BACKEND_DIR / f).stat().st_size == 0]
        if missing_backend:
            self.add_result(CheckResult(
                name="Backend Python Modules",
                category=cat,
                status="FAIL",
                message=f"Missing or empty backend files: {', '.join(missing_backend)}",
                remediation="Verify backend directory contains all required orchestrator modules.",
                critical=True,
            ))
        else:
            self.add_result(CheckResult(
                name="Backend Python Modules",
                category=cat,
                status="PASS",
                message=f"All {len(backend_files)} core backend modules present and non-empty",
                critical=True,
            ))

        frontend_files = [
            "package.json",
            "vite.config.ts",
            "index.html",
            "src/App.tsx",
            "src/main.tsx",
        ]
        missing_frontend = [f for f in frontend_files if not (FRONTEND_DIR / f).is_file() or (FRONTEND_DIR / f).stat().st_size == 0]
        if missing_frontend:
            self.add_result(CheckResult(
                name="Frontend Source Files",
                category=cat,
                status="FAIL",
                message=f"Missing or empty frontend files: {', '.join(missing_frontend)}",
                remediation="Verify frontend source directory contains all React/Vite components.",
                critical=True,
            ))
        else:
            self.add_result(CheckResult(
                name="Frontend Source Files",
                category=cat,
                status="PASS",
                message=f"All {len(frontend_files)} core frontend files present and non-empty",
                critical=True,
            ))

    # ── 3. Environment Variables ──────────────────────────────────────────────
    def check_environment_variables(self):
        self.print_section("Environment Variables & Configuration")
        cat = "Configuration"

        # Check .env existence
        env_files = [
            BACKEND_DIR / "key.env",
            BACKEND_DIR / ".env",
            REPO_ROOT / ".env",
        ]
        existing_env = [f for f in env_files if f.exists()]

        try:
            from dotenv import load_dotenv
            loaded = False
            for ef in existing_env:
                load_dotenv(ef)
                loaded = True
            if not loaded:
                load_dotenv()
            self.add_result(CheckResult(
                name="Environment File Loading",
                category=cat,
                status="PASS" if existing_env else "WARN",
                message=f"Loaded from: {', '.join(str(f.name) for f in existing_env)}" if existing_env else "No .env file found (using system environment)",
                details="Environment variables loaded successfully.",
                critical=False,
            ))
        except ImportError:
            self.add_result(CheckResult(
                name="Environment File Loading",
                category=cat,
                status="FAIL",
                message="python-dotenv is not installed",
                remediation="Run: pip install python-dotenv",
                critical=True,
            ))

        # Check SARVAM_API_KEY
        sarvam_key = os.getenv("SARVAM_API_KEY", "").strip()
        is_placeholder = sarvam_key in ("", "your_sarvam_api_key_here")
        if is_placeholder:
            self.add_result(CheckResult(
                name="Sarvam AI API Key",
                category=cat,
                status="WARN",
                message="SARVAM_API_KEY is not configured or using default placeholder",
                details="Cloud Sarvam STT and Translation fallback will be disabled. Local offline models (Faster-Whisper & IndicTrans2) will be used.",
                remediation="To enable Sarvam Cloud fallback, set a valid key in backend/.env: SARVAM_API_KEY=your_actual_key",
                critical=False,
            ))
        else:
            masked = sarvam_key[:4] + "..." + sarvam_key[-4:] if len(sarvam_key) > 8 else "***"
            self.add_result(CheckResult(
                name="Sarvam AI API Key",
                category=cat,
                status="PASS",
                message=f"Configured ({masked})",
                critical=False,
            ))

        # Check ASR_BACKEND
        asr_backend = os.getenv("ASR_BACKEND", "whisper").lower().strip()
        if asr_backend in ("whisper", "sarvam"):
            self.add_result(CheckResult(
                name="ASR Backend Config",
                category=cat,
                status="PASS",
                message=f"Configured as '{asr_backend}'",
                critical=False,
            ))
        else:
            self.add_result(CheckResult(
                name="ASR Backend Config",
                category=cat,
                status="WARN",
                message=f"Unrecognized ASR_BACKEND '{asr_backend}' (expected 'whisper' or 'sarvam')",
                remediation="Set ASR_BACKEND=whisper in backend/.env",
                critical=False,
            ))

        # Check TRANSLATION_BACKEND
        mt_backend = os.getenv("TRANSLATION_BACKEND", "indictrans2").lower().strip()
        if mt_backend in ("indictrans2", "sarvam", "local"):
            self.add_result(CheckResult(
                name="Translation Backend Config",
                category=cat,
                status="PASS",
                message=f"Configured as '{mt_backend}'",
                critical=False,
            ))
        else:
            self.add_result(CheckResult(
                name="Translation Backend Config",
                category=cat,
                status="WARN",
                message=f"Unrecognized TRANSLATION_BACKEND '{mt_backend}' (expected 'indictrans2' or 'sarvam')",
                remediation="Set TRANSLATION_BACKEND=indictrans2 in backend/.env",
                critical=False,
            ))

    # ── 4. Port Availability ──────────────────────────────────────────────────
    def check_port_availability(self):
        self.print_section("Network & Port Availability")
        cat = "Network"

        def test_port(port: int) -> Tuple[bool, str]:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                try:
                    s.bind(("127.0.0.1", port))
                    return True, "free"
                except OSError:
                    # In use, check if it responds to HTTP
                    return False, "in_use"

        # Backend Port 8000
        free_8000, _ = test_port(8000)
        if free_8000:
            self.add_result(CheckResult(
                name="Backend Port (8000)",
                category=cat,
                status="PASS",
                message="Port 8000 is available for backend service",
                critical=False,
            ))
        else:
            # Check if active IndieTranslate instance is running
            is_indietranslate = False
            try:
                import urllib.request
                req = urllib.request.Request("http://127.0.0.1:8000/health", headers={"User-Agent": "IndieTranslate-Verify"})
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode())
                        if data.get("status") == "ok":
                            is_indietranslate = True
            except Exception:
                pass

            if is_indietranslate:
                self.add_result(CheckResult(
                    name="Backend Port (8000)",
                    category=cat,
                    status="PASS",
                    message="Port 8000 is occupied by an active IndieTranslate backend server (verified via /health)",
                    critical=False,
                ))
            else:
                self.add_result(CheckResult(
                    name="Backend Port (8000)",
                    category=cat,
                    status="WARN",
                    message="Port 8000 is currently occupied by another process",
                    details="When starting uvicorn backend, it will fail unless the occupying process is terminated.",
                    remediation="Identify and stop process on port 8000 (e.g., netstat -ano | findstr :8000)",
                    critical=False,
                ))

        # Frontend Port 5173
        free_5173, _ = test_port(5173)
        if free_5173:
            self.add_result(CheckResult(
                name="Frontend Port (5173)",
                category=cat,
                status="PASS",
                message="Port 5173 is available for Vite frontend development server",
                critical=False,
            ))
        else:
            self.add_result(CheckResult(
                name="Frontend Port (5173)",
                category=cat,
                status="WARN",
                message="Port 5173 is currently occupied (Vite may automatically fallback to 5174)",
                critical=False,
            ))

    # ── 5. Python Dependencies ────────────────────────────────────────────────
    def check_python_dependencies(self):
        self.print_section("Python Dependencies")
        cat = "Python Dependencies"

        # Apply transformers v5 shims before testing IndicTransToolkit
        try:
            import transformers.tokenization_utils
            import transformers.tokenization_utils_base
            if not hasattr(transformers.tokenization_utils, "PreTrainedTokenizerBase"):
                transformers.tokenization_utils.PreTrainedTokenizerBase = transformers.tokenization_utils_base.PreTrainedTokenizerBase
        except Exception:
            pass

        packages_to_check = [
            ("fastapi", "FastAPI Framework", True, "pip install fastapi"),
            ("uvicorn", "Uvicorn ASGI Server", True, "pip install uvicorn"),
            ("websockets", "WebSockets Engine", True, "pip install websockets"),
            ("pydantic", "Pydantic Validation", True, "pip install pydantic"),
            ("torch", "PyTorch ML Engine", True, "pip install torch"),
            ("numpy", "NumPy Matrix Math", True, "pip install numpy"),
            ("faster_whisper", "Faster Whisper (Offline ASR)", True, "pip install faster-whisper"),
            ("ctranslate2", "CTranslate2 Inference Engine", True, "pip install ctranslate2"),
            ("transformers", "Hugging Face Transformers", True, "pip install transformers"),
            ("IndicTransToolkit", "IndicTransToolkit NLP Suite", True, "pip install IndicTransToolkit"),
            ("sarvamai", "Sarvam AI SDK", True, "pip install sarvamai"),
            ("httpx", "HTTPX Async Client", True, "pip install httpx"),
            ("dotenv", "Python Dotenv Config", True, "pip install python-dotenv"),
            ("sentencepiece", "SentencePiece Tokenizer Engine", True, "pip install sentencepiece"),
        ]

        for mod_name, desc, is_critical, install_cmd in packages_to_check:
            t0 = time.perf_counter()
            try:
                mod = __import__(mod_name)
                ver = getattr(mod, "__version__", "installed")
                dur = time.perf_counter() - t0
                self.add_result(CheckResult(
                    name=f"Python: {desc} ({mod_name})",
                    category=cat,
                    status="PASS",
                    message=f"v{ver}",
                    critical=is_critical,
                    duration_s=dur,
                ))
            except ImportError as e:
                dur = time.perf_counter() - t0
                self.add_result(CheckResult(
                    name=f"Python: {desc} ({mod_name})",
                    category=cat,
                    status="FAIL" if is_critical else "WARN",
                    message=f"Not installed ({e})",
                    details=f"Package '{mod_name}' could not be imported.",
                    remediation=f"Run: {install_cmd}",
                    critical=is_critical,
                    duration_s=dur,
                ))

    # ── 6. Node Dependencies ──────────────────────────────────────────────────
    def check_node_dependencies(self):
        self.print_section("Node.js & Frontend Dependencies")
        cat = "Node Dependencies"

        # Check Node.js
        try:
            res_node = subprocess.run(["node", "-v"], capture_output=True, text=True, timeout=5)
            if res_node.returncode == 0:
                node_ver = res_node.stdout.strip()
                self.add_result(CheckResult(
                    name="Node.js Runtime",
                    category=cat,
                    status="PASS",
                    message=f"{node_ver}",
                    critical=True,
                ))
            else:
                self.add_result(CheckResult(
                    name="Node.js Runtime",
                    category=cat,
                    status="FAIL",
                    message=f"node returned error code {res_node.returncode}",
                    remediation="Install or update Node.js (https://nodejs.org/)",
                    critical=True,
                ))
        except FileNotFoundError:
            self.add_result(CheckResult(
                name="Node.js Runtime",
                category=cat,
                status="FAIL",
                message="Node.js executable ('node') not found in PATH",
                remediation="Install Node.js (https://nodejs.org/) and add it to your PATH.",
                critical=True,
            ))
        except Exception as e:
            self.add_result(CheckResult(
                name="Node.js Runtime",
                category=cat,
                status="FAIL",
                message=f"Error checking node: {e}",
                critical=True,
            ))

        # Check npm
        try:
            res_npm = subprocess.run(["npm", "-v"], capture_output=True, text=True, timeout=5, shell=True)
            if res_npm.returncode == 0:
                npm_ver = res_npm.stdout.strip()
                self.add_result(CheckResult(
                    name="npm Package Manager",
                    category=cat,
                    status="PASS",
                    message=f"v{npm_ver}",
                    critical=True,
                ))
            else:
                self.add_result(CheckResult(
                    name="npm Package Manager",
                    category=cat,
                    status="FAIL",
                    message=f"npm returned error code {res_npm.returncode}",
                    remediation="Verify npm installation.",
                    critical=True,
                ))
        except Exception as e:
            self.add_result(CheckResult(
                name="npm Package Manager",
                category=cat,
                status="FAIL",
                message=f"npm not available: {e}",
                remediation="Ensure npm is installed with Node.js.",
                critical=True,
            ))

        # Check frontend/package.json
        pkg_json_path = FRONTEND_DIR / "package.json"
        if not pkg_json_path.exists():
            self.add_result(CheckResult(
                name="Frontend package.json",
                category=cat,
                status="FAIL",
                message="frontend/package.json not found",
                remediation="Ensure frontend/package.json is present.",
                critical=True,
            ))
            return

        try:
            with open(pkg_json_path, "r", encoding="utf-8") as f:
                pkg_data = json.load(f)
            self.add_result(CheckResult(
                name="Frontend package.json Schema",
                category=cat,
                status="PASS",
                message=f"Valid JSON ({pkg_data.get('name', 'frontend')} v{pkg_data.get('version', '0.0.0')})",
                critical=True,
            ))
        except Exception as e:
            self.add_result(CheckResult(
                name="Frontend package.json Schema",
                category=cat,
                status="FAIL",
                message=f"Invalid package.json: {e}",
                critical=True,
            ))
            return

        # Check node_modules
        node_modules_dir = FRONTEND_DIR / "node_modules"
        if not node_modules_dir.is_dir():
            self.add_result(CheckResult(
                name="Frontend node_modules",
                category=cat,
                status="FAIL",
                message="frontend/node_modules directory is missing",
                remediation="Run: cd frontend && npm install",
                critical=True,
            ))
            return

        # Check critical node packages
        critical_node_pkgs = ["react", "react-dom", "vite", "typescript", "@vitejs/plugin-react"]
        missing_pkgs = []
        for pkg in critical_node_pkgs:
            pkg_p = node_modules_dir / pkg
            if not pkg_p.exists():
                missing_pkgs.append(pkg)

        if missing_pkgs:
            self.add_result(CheckResult(
                name="Frontend Installed Dependencies",
                category=cat,
                status="FAIL",
                message=f"Missing packages in node_modules: {', '.join(missing_pkgs)}",
                remediation="Run: cd frontend && npm install",
                critical=True,
            ))
        else:
            self.add_result(CheckResult(
                name="Frontend Installed Dependencies",
                category=cat,
                status="PASS",
                message=f"All {len(critical_node_pkgs)} critical frontend packages present in node_modules",
                critical=True,
            ))

    # ── 7. Faster Whisper Model Integrity ─────────────────────────────────────
    def check_faster_whisper_model_integrity(self) -> Path:
        self.print_section("Faster-Whisper Model Integrity")
        cat = "Faster-Whisper Model"

        # Resolve model directory
        candidate_paths = [
            MODELS_DIR / "faster-whisper-small",
            MODELS_DIR / "faster-whisper-small" / "models--Systran--faster-whisper-small" / "snapshots" / "536b0662742c02347bc0e980a01041f333bce120",
        ]
        whisper_dir = None
        for p in candidate_paths:
            if p.exists() and (p / "model.bin").is_file():
                whisper_dir = p
                break
        if whisper_dir is None:
            whisper_dir = MODELS_DIR / "faster-whisper-small"

        if not whisper_dir.exists():
            self.add_result(CheckResult(
                name="Faster-Whisper Directory",
                category=cat,
                status="FAIL",
                message=f"Directory not found: {whisper_dir}",
                remediation="Download or link faster-whisper-small into models/faster-whisper-small",
                critical=True,
            ))
            return whisper_dir

        # Check model.bin
        bin_path = whisper_dir / "model.bin"
        if not bin_path.exists():
            self.add_result(CheckResult(
                name="Whisper Weights (model.bin)",
                category=cat,
                status="FAIL",
                message="model.bin missing",
                remediation="Ensure model.bin is downloaded in models/faster-whisper-small",
                critical=True,
            ))
        else:
            size_mb = round(bin_path.stat().st_size / (1024 * 1024), 2)
            if size_mb < 100:
                self.add_result(CheckResult(
                    name="Whisper Weights (model.bin)",
                    category=cat,
                    status="FAIL",
                    message=f"model.bin is unexpectedly small ({size_mb} MB < 100 MB)",
                    critical=True,
                ))
            else:
                self.add_result(CheckResult(
                    name="Whisper Weights (model.bin)",
                    category=cat,
                    status="PASS",
                    message=f"Present ({size_mb} MB)",
                    critical=True,
                ))

        # Check config.json
        cfg_path = whisper_dir / "config.json"
        if not cfg_path.exists():
            self.add_result(CheckResult(
                name="Whisper Config (config.json)",
                category=cat,
                status="FAIL",
                message="config.json missing",
                critical=True,
            ))
        else:
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_json = json.load(f)
                self.add_result(CheckResult(
                    name="Whisper Config (config.json)",
                    category=cat,
                    status="PASS",
                    message="Valid JSON configuration",
                    critical=True,
                ))
            except Exception as e:
                self.add_result(CheckResult(
                    name="Whisper Config (config.json)",
                    category=cat,
                    status="FAIL",
                    message=f"Corrupted config.json: {e}",
                    critical=True,
                ))

        # Check tokenizer.json
        tok_path = whisper_dir / "tokenizer.json"
        if not tok_path.exists():
            self.add_result(CheckResult(
                name="Whisper Tokenizer (tokenizer.json)",
                category=cat,
                status="FAIL",
                message="tokenizer.json missing",
                critical=True,
            ))
        else:
            try:
                with open(tok_path, "r", encoding="utf-8") as f:
                    tok_json = json.load(f)
                vocab_len = len(tok_json.get("model", {}).get("vocab", {}))
                self.add_result(CheckResult(
                    name="Whisper Tokenizer (tokenizer.json)",
                    category=cat,
                    status="PASS",
                    message=f"Valid JSON ({vocab_len} vocab entries)",
                    critical=True,
                ))
            except Exception as e:
                self.add_result(CheckResult(
                    name="Whisper Tokenizer (tokenizer.json)",
                    category=cat,
                    status="FAIL",
                    message=f"Corrupted tokenizer.json: {e}",
                    critical=True,
                ))

        # Check vocabulary.txt
        voc_path = whisper_dir / "vocabulary.txt"
        if not voc_path.exists():
            self.add_result(CheckResult(
                name="Whisper Vocabulary (vocabulary.txt)",
                category=cat,
                status="FAIL",
                message="vocabulary.txt missing",
                critical=True,
            ))
        else:
            try:
                with open(voc_path, "r", encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)
                self.add_result(CheckResult(
                    name="Whisper Vocabulary (vocabulary.txt)",
                    category=cat,
                    status="PASS",
                    message=f"Present ({line_count} tokens)",
                    critical=True,
                ))
            except Exception as e:
                self.add_result(CheckResult(
                    name="Whisper Vocabulary (vocabulary.txt)",
                    category=cat,
                    status="FAIL",
                    message=f"Corrupted vocabulary.txt: {e}",
                    critical=True,
                ))

        return whisper_dir

    # ── 8. IndicTrans2 Model Integrity ────────────────────────────────────────
    def check_indictrans2_model_integrity(self) -> Path:
        self.print_section("IndicTrans2 Model Integrity")
        cat = "IndicTrans2 Model"

        indic_dir = MODELS_DIR / "indictrans2-indic-indic-320M"
        if not indic_dir.exists():
            self.add_result(CheckResult(
                name="IndicTrans2 Directory",
                category=cat,
                status="FAIL",
                message=f"Directory not found: {indic_dir}",
                remediation="Download or link indictrans2-indic-indic-320M in models/",
                critical=True,
            ))
            return indic_dir

        # Check weights (model.safetensors or pytorch_model.bin)
        sf_path = indic_dir / "model.safetensors"
        bin_path = indic_dir / "pytorch_model.bin"
        if sf_path.exists():
            size_mb = round(sf_path.stat().st_size / (1024 * 1024), 2)
            try:
                import safetensors
                meta = safetensors.safe_open(str(sf_path), framework="pt")
                tensor_count = len(meta.keys())
                self.add_result(CheckResult(
                    name="IndicTrans2 Weights (model.safetensors)",
                    category=cat,
                    status="PASS",
                    message=f"Valid Safetensors ({size_mb} MB, {tensor_count} tensors)",
                    critical=True,
                ))
            except ImportError:
                self.add_result(CheckResult(
                    name="IndicTrans2 Weights (model.safetensors)",
                    category=cat,
                    status="PASS",
                    message=f"Present ({size_mb} MB)",
                    critical=True,
                ))
            except Exception as e:
                self.add_result(CheckResult(
                    name="IndicTrans2 Weights (model.safetensors)",
                    category=cat,
                    status="FAIL",
                    message=f"Corrupted safetensors file: {e}",
                    critical=True,
                ))
        elif bin_path.exists():
            size_mb = round(bin_path.stat().st_size / (1024 * 1024), 2)
            self.add_result(CheckResult(
                name="IndicTrans2 Weights (pytorch_model.bin)",
                category=cat,
                status="PASS",
                message=f"Present ({size_mb} MB)",
                critical=True,
            ))
        else:
            self.add_result(CheckResult(
                name="IndicTrans2 Weights",
                category=cat,
                status="FAIL",
                message="Neither model.safetensors nor pytorch_model.bin found",
                critical=True,
            ))

        # Check config.json
        cfg_path = indic_dir / "config.json"
        if not cfg_path.exists():
            self.add_result(CheckResult(
                name="IndicTrans2 Config (config.json)",
                category=cat,
                status="FAIL",
                message="config.json missing",
                critical=True,
            ))
        else:
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                mtype = cfg_data.get("model_type", "IndicTrans")
                vsize = cfg_data.get("vocab_size", 0)
                self.add_result(CheckResult(
                    name="IndicTrans2 Config (config.json)",
                    category=cat,
                    status="PASS",
                    message=f"Valid JSON (model_type='{mtype}', vocab_size={vsize})",
                    critical=True,
                ))
            except Exception as e:
                self.add_result(CheckResult(
                    name="IndicTrans2 Config (config.json)",
                    category=cat,
                    status="FAIL",
                    message=f"Corrupted config.json: {e}",
                    critical=True,
                ))

        # Check vocab files: dict.SRC.json and dict.TGT.json
        for fname, label in [("dict.SRC.json", "Source Vocabulary"), ("dict.TGT.json", "Target Vocabulary")]:
            fp = indic_dir / fname
            if not fp.exists():
                self.add_result(CheckResult(
                    name=f"IndicTrans2 {label} ({fname})",
                    category=cat,
                    status="FAIL",
                    message=f"{fname} missing",
                    critical=True,
                ))
            else:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        vdata = json.load(f)
                    self.add_result(CheckResult(
                        name=f"IndicTrans2 {label} ({fname})",
                        category=cat,
                        status="PASS",
                        message=f"Valid JSON ({len(vdata)} tokens)",
                        critical=True,
                    ))
                except Exception as e:
                    self.add_result(CheckResult(
                        name=f"IndicTrans2 {label} ({fname})",
                        category=cat,
                        status="FAIL",
                        message=f"Corrupted {fname}: {e}",
                        critical=True,
                    ))

        # Check SPM model files: model.SRC and model.TGT
        for fname, label in [("model.SRC", "SentencePiece Source Model"), ("model.TGT", "SentencePiece Target Model")]:
            fp = indic_dir / fname
            if not fp.exists():
                self.add_result(CheckResult(
                    name=f"IndicTrans2 {label} ({fname})",
                    category=cat,
                    status="FAIL",
                    message=f"{fname} missing",
                    critical=True,
                ))
            else:
                sz_mb = round(fp.stat().st_size / (1024 * 1024), 2)
                self.add_result(CheckResult(
                    name=f"IndicTrans2 {label} ({fname})",
                    category=cat,
                    status="PASS",
                    message=f"Present ({sz_mb} MB)",
                    critical=True,
                ))

        # Check dynamic modeling scripts
        dyn_files = [
            ("configuration_indictrans.py", 1000),
            ("modeling_indictrans.py", 5000),
            ("tokenization_indictrans.py", 1000),
        ]
        for df, min_sz in dyn_files:
            fp = indic_dir / df
            if not fp.exists() or fp.stat().st_size < min_sz:
                self.add_result(CheckResult(
                    name=f"Dynamic Module: {df}",
                    category=cat,
                    status="FAIL",
                    message=f"Missing or corrupted {df}",
                    critical=True,
                ))
            else:
                self.add_result(CheckResult(
                    name=f"Dynamic Module: {df}",
                    category=cat,
                    status="PASS",
                    message=f"Present ({round(fp.stat().st_size/1024, 1)} KB)",
                    critical=True,
                ))

        return indic_dir

    # ── 9. Tokenizer Loading ──────────────────────────────────────────────────
    def check_tokenizer_loading(self, whisper_dir: Path, indic_dir: Path):
        self.print_section("Tokenizer Loading")
        cat = "Tokenizer Loading"

        # 1. Faster Whisper Tokenizer
        t0 = time.perf_counter()
        try:
            tok_p = whisper_dir / "tokenizer.json"
            voc_p = whisper_dir / "vocabulary.txt"
            if tok_p.exists() and voc_p.exists():
                with open(tok_p, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                vocab_size = len(tok_data.get("model", {}).get("vocab", {}))
                dur = time.perf_counter() - t0
                self.add_result(CheckResult(
                    name="Faster-Whisper Tokenizer",
                    category=cat,
                    status="PASS",
                    message=f"Verified vocabulary structure ({vocab_size} tokens)",
                    critical=True,
                    duration_s=dur,
                ))
            else:
                self.add_result(CheckResult(
                    name="Faster-Whisper Tokenizer",
                    category=cat,
                    status="FAIL",
                    message="Missing tokenizer.json or vocabulary.txt",
                    critical=True,
                ))
        except Exception as e:
            self.add_result(CheckResult(
                name="Faster-Whisper Tokenizer",
                category=cat,
                status="FAIL",
                message=f"Failed loading Whisper tokenizer: {e}",
                critical=True,
            ))

        # 2. IndicTrans2 Tokenizer
        t0 = time.perf_counter()
        try:
            # Check if sentencepiece is installed
            import sentencepiece
        except ImportError:
            self.add_result(CheckResult(
                name="IndicTrans2 Tokenizer",
                category=cat,
                status="FAIL",
                message="Cannot load tokenizer: 'sentencepiece' package is missing",
                details="IndicTrans2 uses SentencePieceProcessor in tokenization_indictrans.py.",
                remediation="Run: pip install sentencepiece",
                critical=True,
            ))
            return

        try:
            # Apply runtime transformers v5 shims for IndicTrans2 dynamic tokenizer
            import types
            if "transformers.onnx" not in sys.modules:
                m_onnx = types.ModuleType("transformers.onnx")
                m_onnx.OnnxConfig = object
                m_onnx.OnnxSeq2SeqConfigWithPast = object
                m_onnx_utils = types.ModuleType("transformers.onnx.utils")
                m_onnx_utils.compute_effective_axis_dimension = lambda *a, **k: None
                m_onnx.utils = m_onnx_utils
                sys.modules["transformers.onnx"] = m_onnx
                sys.modules["transformers.onnx.utils"] = m_onnx_utils

            import transformers.tokenization_utils
            import transformers.tokenization_utils_base
            if not hasattr(transformers.tokenization_utils, "PreTrainedTokenizerBase"):
                transformers.tokenization_utils.PreTrainedTokenizerBase = transformers.tokenization_utils_base.PreTrainedTokenizerBase

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(indic_dir),
                trust_remote_code=True,
                local_files_only=True,
            )
            dur = time.perf_counter() - t0
            self.add_result(CheckResult(
                name="IndicTrans2 Tokenizer",
                category=cat,
                status="PASS",
                message=f"Loaded successfully ({type(tokenizer).__name__})",
                critical=True,
                duration_s=dur,
            ))
        except Exception as e:
            dur = time.perf_counter() - t0
            self.add_result(CheckResult(
                name="IndicTrans2 Tokenizer",
                category=cat,
                status="FAIL",
                message=f"Failed loading IndicTrans2 tokenizer: {e}",
                details=str(e),
                remediation="Ensure sentencepiece is installed and model directory is complete.",
                critical=True,
                duration_s=dur,
            ))

    # ── 10. Model Loading Time ────────────────────────────────────────────────
    def check_model_loading_time(self, whisper_dir: Path, indic_dir: Path):
        self.print_section("Model Loading Time & Inference Benchmark")
        cat = "Model Loading"

        if self.args.skip_model_load:
            self.add_result(CheckResult(
                name="Faster-Whisper Model Load",
                category=cat,
                status="SKIP",
                message="Skipped (--skip-model-load)",
                critical=False,
            ))
            self.add_result(CheckResult(
                name="IndicTrans2 Model Load",
                category=cat,
                status="SKIP",
                message="Skipped (--skip-model-load)",
                critical=False,
            ))
            return

        # 1. Benchmark Faster Whisper
        try:
            import numpy as np
            import torch
            from faster_whisper import WhisperModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            t0 = time.perf_counter()
            whisper_model = WhisperModel(
                str(whisper_dir),
                device=device,
                compute_type=compute_type,
                cpu_threads=4,
            )
            load_dur = time.perf_counter() - t0

            # Test inference benchmark on 1 second of dummy audio
            t_inf_0 = time.perf_counter()
            dummy_pcm = np.zeros(16000, dtype=np.float32)
            segments, info = whisper_model.transcribe(
                dummy_pcm,
                language="en",
                beam_size=1,
                temperature=0.0,
            )
            list(segments)  # evaluate generator
            inf_dur = time.perf_counter() - t_inf_0

            self.add_result(CheckResult(
                name="Faster-Whisper Model Loading & Inference",
                category=cat,
                status="PASS",
                message=f"Loaded in {load_dur:.2f}s on {device} ({compute_type}) | Inference benchmark: {inf_dur*1000:.1f}ms",
                critical=True,
                duration_s=load_dur,
            ))
        except Exception as e:
            self.add_result(CheckResult(
                name="Faster-Whisper Model Loading",
                category=cat,
                status="FAIL",
                message=f"Failed to load Faster-Whisper: {e}",
                remediation="Verify faster-whisper package and model weights integrity in models/faster-whisper-small.",
                critical=True,
            ))

        # 2. Benchmark IndicTrans2 Model Loading
        try:
            import types
            import torch
            from transformers import AutoConfig, AutoModelForSeq2SeqLM
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            # Transformers v5 compatibility shims
            if "transformers.onnx" not in sys.modules:
                m_onnx = types.ModuleType("transformers.onnx")
                m_onnx.OnnxConfig = object
                m_onnx.OnnxSeq2SeqConfigWithPast = object
                m_onnx_utils = types.ModuleType("transformers.onnx.utils")
                m_onnx_utils.compute_effective_axis_dimension = lambda *a, **k: None
                m_onnx.utils = m_onnx_utils
                sys.modules["transformers.onnx"] = m_onnx
                sys.modules["transformers.onnx.utils"] = m_onnx_utils

            # Handle tie_weights keyword argument in Transformers v5
            model_class = get_class_from_dynamic_module(
                "modeling_indictrans.IndicTransForConditionalGeneration",
                str(indic_dir),
            )
            orig_tie_weights = model_class.tie_weights

            def safe_tie_weights(self, *args, **kwargs):
                return orig_tie_weights(self)

            model_class.tie_weights = safe_tie_weights

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            t0 = time.perf_counter()
            indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                str(indic_dir),
                trust_remote_code=True,
                local_files_only=True,
                dtype=dtype,
            ).to(device)
            load_dur = time.perf_counter() - t0

            param_count_m = round(sum(p.numel() for p in indic_model.parameters()) / 1e6, 1)

            self.add_result(CheckResult(
                name="IndicTrans2 Model Loading",
                category=cat,
                status="PASS",
                message=f"Loaded in {load_dur:.2f}s on {device} ({param_count_m}M parameters)",
                critical=True,
                duration_s=load_dur,
            ))
        except Exception as e:
            self.add_result(CheckResult(
                name="IndicTrans2 Model Loading",
                category=cat,
                status="FAIL",
                message=f"Failed to load IndicTrans2 model: {e}",
                details=str(e),
                critical=True,
            ))

    # ── 11. API Startup ───────────────────────────────────────────────────────
    def check_api_startup(self) -> Any:
        self.print_section("FastAPI Application Startup")
        cat = "API Startup"

        try:
            t0 = time.perf_counter()
            from backend.main import app
            from fastapi import FastAPI

            if not isinstance(app, FastAPI):
                self.add_result(CheckResult(
                    name="FastAPI App Instance",
                    category=cat,
                    status="FAIL",
                    message="backend.main.app is not a FastAPI instance",
                    critical=True,
                ))
                return None

            dur = time.perf_counter() - t0
            routes = [r.path for r in app.routes if hasattr(r, "path")]
            expected_routes = [
                "/health",
                "/api/info",
                "/api/session/create",
                "/api/session/{session_id}",
                "/ws/transcribe",
                "/ws/session",
            ]
            missing_routes = [r for r in expected_routes if r not in routes]

            if missing_routes:
                self.add_result(CheckResult(
                    name="FastAPI Route Registration",
                    category=cat,
                    status="FAIL",
                    message=f"Missing routes: {', '.join(missing_routes)}",
                    critical=True,
                ))
            else:
                self.add_result(CheckResult(
                    name="FastAPI App Initialization",
                    category=cat,
                    status="PASS",
                    message=f"Successfully initialized (Title: '{app.title}', {len(routes)} routes registered)",
                    critical=True,
                    duration_s=dur,
                ))

            return app
        except Exception as e:
            self.add_result(CheckResult(
                name="FastAPI App Initialization",
                category=cat,
                status="FAIL",
                message=f"Failed to import backend.main.app: {e}",
                details=str(e),
                critical=True,
            ))
            return None

    # ── 12. OpenAPI Endpoint ──────────────────────────────────────────────────
    def check_openapi_endpoint(self, app: Any):
        self.print_section("OpenAPI Specification")
        cat = "OpenAPI"

        if app is None:
            self.add_result(CheckResult(
                name="OpenAPI Endpoint (/openapi.json)",
                category=cat,
                status="SKIP",
                message="Skipped because API startup failed",
                critical=True,
            ))
            return

        try:
            from fastapi.testclient import TestClient
            client = TestClient(app)

            # 1. /openapi.json
            t0 = time.perf_counter()
            res = client.get("/openapi.json")
            dur = time.perf_counter() - t0

            if res.status_code != 200:
                self.add_result(CheckResult(
                    name="OpenAPI Schema (/openapi.json)",
                    category=cat,
                    status="FAIL",
                    message=f"Status code {res.status_code} (expected 200)",
                    critical=True,
                ))
                return

            schema = res.json()
            openapi_ver = schema.get("openapi", "unknown")
            info_title = schema.get("info", {}).get("title", "")
            path_count = len(schema.get("paths", {}))

            self.add_result(CheckResult(
                name="OpenAPI Schema (/openapi.json)",
                category=cat,
                status="PASS",
                message=f"Status 200 OK (OpenAPI {openapi_ver}, '{info_title}', {path_count} paths)",
                critical=True,
                duration_s=dur,
            ))

            # 2. /docs Swagger UI
            res_docs = client.get("/docs")
            if res_docs.status_code == 200:
                self.add_result(CheckResult(
                    name="Swagger Documentation (/docs)",
                    category=cat,
                    status="PASS",
                    message="Status 200 OK (Interactive Swagger UI available)",
                    critical=False,
                ))
            else:
                self.add_result(CheckResult(
                    name="Swagger Documentation (/docs)",
                    category=cat,
                    status="WARN",
                    message=f"Status code {res_docs.status_code} on /docs",
                    critical=False,
                ))
        except Exception as e:
            self.add_result(CheckResult(
                name="OpenAPI Schema (/openapi.json)",
                category=cat,
                status="FAIL",
                message=f"Failed requesting OpenAPI schema: {e}",
                critical=True,
            ))

    # ── 13. Health Endpoint ───────────────────────────────────────────────────
    def check_health_endpoint(self, app: Any):
        self.print_section("Health & Diagnostics Endpoint")
        cat = "Health Endpoint"

        if app is None:
            self.add_result(CheckResult(
                name="Health Endpoint (/health)",
                category=cat,
                status="SKIP",
                message="Skipped because API startup failed",
                critical=True,
            ))
            return

        try:
            from fastapi.testclient import TestClient
            client = TestClient(app)

            t0 = time.perf_counter()
            res = client.get("/health")
            dur = time.perf_counter() - t0

            if res.status_code != 200:
                self.add_result(CheckResult(
                    name="Health Endpoint (/health)",
                    category=cat,
                    status="FAIL",
                    message=f"Status code {res.status_code} (expected 200)",
                    critical=True,
                ))
                return

            payload = res.json()
            required_keys = ["status", "asr_backend", "translation_backend", "sarvam_configured", "local_ip"]
            missing_keys = [k for k in required_keys if k not in payload]

            if missing_keys:
                self.add_result(CheckResult(
                    name="Health Endpoint Schema",
                    category=cat,
                    status="FAIL",
                    message=f"Missing expected keys: {', '.join(missing_keys)}",
                    critical=True,
                ))
            elif payload.get("status") != "ok":
                self.add_result(CheckResult(
                    name="Health Endpoint Status",
                    category=cat,
                    status="FAIL",
                    message=f"status = '{payload.get('status')}' (expected 'ok')",
                    critical=True,
                ))
            else:
                asr = payload.get("asr_backend")
                mt = payload.get("translation_backend")
                sarvam = payload.get("sarvam_configured")
                ip = payload.get("local_ip")
                self.add_result(CheckResult(
                    name="Health Endpoint (/health)",
                    category=cat,
                    status="PASS",
                    message=f"Status 200 OK [status='ok', asr='{asr}', mt='{mt}', sarvam_configured={sarvam}, ip='{ip}']",
                    critical=True,
                    duration_s=dur,
                ))

            # Test /api/info
            res_info = client.get("/api/info")
            if res_info.status_code == 200:
                self.add_result(CheckResult(
                    name="System Info Endpoint (/api/info)",
                    category=cat,
                    status="PASS",
                    message="Status 200 OK (Connection metadata endpoint verified)",
                    critical=False,
                ))
            else:
                self.add_result(CheckResult(
                    name="System Info Endpoint (/api/info)",
                    category=cat,
                    status="WARN",
                    message=f"Status code {res_info.status_code}",
                    critical=False,
                ))
        except Exception as e:
            self.add_result(CheckResult(
                name="Health Endpoint (/health)",
                category=cat,
                status="FAIL",
                message=f"Health endpoint check failed: {e}",
                critical=True,
            ))

    # ── 14. WebSocket Handshake ───────────────────────────────────────────────
    def check_websocket_handshake(self, app: Any):
        self.print_section("WebSocket Handshake & Protocol")
        cat = "WebSocket Protocol"

        if self.args.skip_ws:
            self.add_result(CheckResult(
                name="WebSocket Handshake (/ws/session)",
                category=cat,
                status="SKIP",
                message="Skipped (--skip-ws)",
                critical=False,
            ))
            return

        if app is None:
            self.add_result(CheckResult(
                name="WebSocket Handshake (/ws/session)",
                category=cat,
                status="SKIP",
                message="Skipped because API startup failed",
                critical=True,
            ))
            return

        try:
            from fastapi.testclient import TestClient
            client = TestClient(app)

            t0 = time.perf_counter()
            with client.websocket_connect("/ws/session") as ws:
                # Send create_session command
                ws.send_json({"action": "create_session", "language": "ta-IN"})
                resp = ws.receive_json()
                dur = time.perf_counter() - t0

                if resp.get("type") == "session.created" and resp.get("session_id"):
                    session_id = resp.get("session_id")
                    pid = resp.get("participant_id")
                    self.add_result(CheckResult(
                        name="WebSocket Session Handshake (/ws/session)",
                        category=cat,
                        status="PASS",
                        message=f"Handshake passed, session created: '{session_id}' (Participant: {pid})",
                        critical=True,
                        duration_s=dur,
                    ))
                else:
                    self.add_result(CheckResult(
                        name="WebSocket Session Handshake (/ws/session)",
                        category=cat,
                        status="FAIL",
                        message=f"Unexpected response payload: {resp}",
                        critical=True,
                    ))
        except Exception as e:
            self.add_result(CheckResult(
                name="WebSocket Session Handshake (/ws/session)",
                category=cat,
                status="FAIL",
                message=f"WebSocket connection failed: {e}",
                details=str(e),
                remediation="Verify websockets package and backend/main.py WebSocket routes.",
                critical=True,
            ))

    # ── Main Run Orchestrator ─────────────────────────────────────────────────
    def run(self) -> int:
        self.t_start = time.perf_counter()
        c = self.colors

        if not self.args.json:
            print(f"\n{c.BOLD}{'=' * 80}{c.RESET}")
            print(f"{c.BOLD}{c.CYAN}{'INDIETRANSLATE SYSTEM VERIFICATION SUITE'.center(80)}{c.RESET}")
            print(f"{c.BOLD}{'=' * 80}{c.RESET}")
            print(f"Timestamp:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Python:      {sys.version.split()[0]} ({sys.executable})")
            print(f"Directory:   {REPO_ROOT}")
            print(f"{'-' * 80}")

        # Run all verification phases
        self.check_hardware_and_platform()
        self.check_required_folders_and_files()
        self.check_environment_variables()
        self.check_port_availability()
        self.check_python_dependencies()
        self.check_node_dependencies()
        whisper_dir = self.check_faster_whisper_model_integrity()
        indic_dir = self.check_indictrans2_model_integrity()
        self.check_tokenizer_loading(whisper_dir, indic_dir)
        self.check_model_loading_time(whisper_dir, indic_dir)
        app = self.check_api_startup()
        self.check_openapi_endpoint(app)
        self.check_health_endpoint(app)
        self.check_websocket_handshake(app)

        total_dur = time.perf_counter() - self.t_start

        # Compute summary metrics
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed_crit = sum(1 for r in self.results if r.status == "FAIL" and r.critical)
        failed_noncrit = sum(1 for r in self.results if r.status == "FAIL" and not r.critical)
        warnings = sum(1 for r in self.results if r.status == "WARN")
        skipped = sum(1 for r in self.results if r.status == "SKIP")

        # JSON Output Mode
        if self.args.json:
            output = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(total_dur, 3),
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed_critical": failed_crit,
                    "failed_noncritical": failed_noncrit,
                    "warnings": warnings,
                    "skipped": skipped,
                    "success": (failed_crit == 0),
                },
                "checks": [r.to_dict() for r in self.results],
            }
            print(json.dumps(output, indent=2))
            return 1 if failed_crit > 0 else 0

        # Formatted Terminal Report
        print(f"\n{c.BOLD}{'=' * 80}{c.RESET}")
        print(f"{c.BOLD}{'CHECKLIST SUMMARY'.center(80)}{c.RESET}")
        print(f"{c.BOLD}{'=' * 80}{c.RESET}")

        current_cat = ""
        for r in self.results:
            if r.category != current_cat:
                current_cat = r.category
                print(f"\n{c.BOLD}{c.CYAN}--- {current_cat} ---{c.RESET}")

            if r.status == "PASS":
                status_badge = f"{c.GREEN}{c.BOLD}[PASS]{c.RESET}"
            elif r.status == "FAIL":
                status_badge = f"{c.RED}{c.BOLD}[FAIL]{c.RESET}"
            elif r.status == "WARN":
                status_badge = f"{c.YELLOW}{c.BOLD}[WARN]{c.RESET}"
            elif r.status == "SKIP":
                status_badge = f"{c.BLUE}{c.BOLD}[SKIP]{c.RESET}"
            else:
                status_badge = f"{c.CYAN}{c.BOLD}[INFO]{c.RESET}"

            name_col = f"{r.name[:38]:<38}"
            msg_col = r.message
            if r.duration_s > 0.05:
                msg_col += f" {c.DIM}({r.duration_s:.2f}s){c.RESET}"

            print(f"  {status_badge} {name_col} {msg_col}")

        # Summary Banner
        print(f"\n{c.BOLD}{'=' * 80}{c.RESET}")
        print(f"{c.BOLD}{'FINAL STATUS & SCORECARD'.center(80)}{c.RESET}")
        print(f"{c.BOLD}{'=' * 80}{c.RESET}")

        print(f"  Total Checks:       {total}")
        print(f"  {c.GREEN}Passed:             {passed}{c.RESET}")
        if failed_crit > 0:
            print(f"  {c.RED}Failed (Critical):  {failed_crit}{c.RESET}")
        if warnings > 0:
            print(f"  {c.YELLOW}Warnings:           {warnings}{c.RESET}")
        if skipped > 0:
            print(f"  {c.BLUE}Skipped:            {skipped}{c.RESET}")
        print(f"  Total Duration:     {total_dur:.2f}s")
        print(f"{'-' * 80}")

        # Remediation Section
        issues = [r for r in self.results if r.status in ("FAIL", "WARN")]
        if issues:
            print(f"\n{c.BOLD}{c.YELLOW}ACTIONABLE DIAGNOSTICS & REMEDIATION STEPS:{c.RESET}")
            for idx, item in enumerate(issues, 1):
                badge = f"{c.RED}[FAIL]{c.RESET}" if item.status == "FAIL" else f"{c.YELLOW}[WARN]{c.RESET}"
                crit_tag = " (Critical)" if item.critical else ""
                print(f"\n  {idx}. {badge} {c.BOLD}{item.name}{crit_tag}{c.RESET}")
                if item.message:
                    print(f"     Issue:       {item.message}")
                if item.details:
                    print(f"     Details:     {item.details}")
                if item.remediation:
                    print(f"     {c.CYAN}Remediation: {item.remediation}{c.RESET}")

        print(f"\n{c.BOLD}{'=' * 80}{c.RESET}")
        if failed_crit == 0:
            print(f"{c.BOLD}{c.GREEN}{'SUCCESS: ALL CRITICAL VERIFICATION CHECKS PASSED!'.center(80)}{c.RESET}")
            print(f"{c.BOLD}{'=' * 80}\n{c.RESET}")
            return 0
        else:
            print(f"{c.BOLD}{c.RED}{f'FAILURE: {failed_crit} CRITICAL CHECK(S) FAILED. EXITING WITH STATUS 1.'.center(80)}{c.RESET}")
            print(f"{c.BOLD}{'=' * 80}\n{c.RESET}")
            return 1


# ── Automatic Virtualenv Switcher & Entry Point ───────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Non-Destructive Verification Suite for IndieTranslate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip heavy neural model loading benchmarks (faster test execution)",
    )
    parser.add_argument(
        "--skip-ws",
        action="store_true",
        help="Skip WebSocket handshake test",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colorized output",
    )
    parser.add_argument(
        "--no-venv-switch",
        action="store_true",
        help="Do not automatically re-execute inside backend/.venv if detected",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as structured JSON",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose debug logs and stack traces",
    )

    args = parser.parse_args()

    # Automatic project virtualenv detection:
    # If the user runs `python verify_project.py` outside the venv, and backend/.venv exists,
    # re-execute using the project's dedicated virtual environment python executable.
    venv_python_win = BACKEND_DIR / ".venv" / "Scripts" / "python.exe"
    venv_python_nix = BACKEND_DIR / ".venv" / "bin" / "python"
    venv_python = venv_python_win if venv_python_win.exists() else venv_python_nix

    if (
        venv_python.exists()
        and not args.no_venv_switch
        and os.environ.get("INDIETRANSLATE_VERIFY_REEXEC") != "1"
    ):
        current_exe = Path(sys.executable).resolve()
        target_exe = venv_python.resolve()
        if current_exe != target_exe:
            # Check if current interpreter lacks faster-whisper or other core deps
            try:
                import faster_whisper
            except ImportError:
                if not args.json:
                    print(f"[*] Detected project virtualenv at {venv_python}. Switching to project environment...")
                env = os.environ.copy()
                env["INDIETRANSLATE_VERIFY_REEXEC"] = "1"
                ret = subprocess.call([str(venv_python)] + sys.argv, env=env)
                sys.exit(ret)

    suite = VerificationSuite(args)
    exit_code = suite.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
