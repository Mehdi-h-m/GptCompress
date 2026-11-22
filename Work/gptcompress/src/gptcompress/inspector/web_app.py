"""Local inspection app. Run with the Work virtual environment Python."""
import argparse
import json
import socket
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gptcompress.inspector.inspection import (STORE, add_version, benchmark_deltas, discover, matrix_compare,
                                               matrix_inspect, read_json, versions)

JOBS = {}
JOBS_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)
STATIC = Path(__file__).parent / "static"
ACTIVE = {"queued", "running", "cancelling"}


class LocalServer(ThreadingHTTPServer):
    # Windows allows competing listeners with SO_REUSEADDR. Keep each port
    # attached to exactly one inspector process and Python environment.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def submit(request):
    kind = request.get("kind")
    if kind not in {"matrix", "matrix_compare", "record", "import"}:
        raise ValueError("Unknown job kind")
    identifier = uuid.uuid4().hex[:12]
    cancelled = threading.Event()
    job = {"id": identifier, "kind": kind, "status": "queued", "progress": 0,
           "message": "Waiting for inspection worker", "cancel": cancelled}
    with JOBS_LOCK:
        if sum(item["status"] in ACTIVE for item in JOBS.values()) >= 4:
            raise ValueError("Inspection queue is full. Let a job finish or cancel one.")
        while len(JOBS) >= 24:
            oldest = next(key for key, item in JOBS.items() if item["status"] not in ACTIVE)
            del JOBS[oldest]
        JOBS[identifier] = job

    def progress(message, fraction):
        with JOBS_LOCK:
            job.update(message=message, progress=fraction)

    def work():
        try:
            with JOBS_LOCK:
                if cancelled.is_set():
                    raise InterruptedError("Cancelled")
                job["status"] = "running"
            if kind == "matrix":
                result = matrix_inspect(request, progress, cancelled.is_set)
            elif kind == "matrix_compare":
                result = matrix_compare(request, progress, cancelled.is_set)
            elif kind == "import":
                progress("Reading checkpoint inventory and verifying identity", .2)
                result = add_version(request["path"], request.get("name"), cancelled.is_set)
            else:
                progress("Preparing activation recorder", 0)
        from gptcompress.inspector.recording import record
                result = record(request, progress, cancelled.is_set)
            with JOBS_LOCK:
                job.update(status="complete", progress=1, message="Finished",
                           result={"recording_id": result["id"]} if kind == "record" else result)
        except Exception as error:
            with JOBS_LOCK:
                status = "cancelled" if isinstance(error, InterruptedError) else "failed"
                job.update(status=status, message="Cancelled" if status == "cancelled" else "Job failed",
                           error=str(error))
    EXECUTOR.submit(work)
    return {"id": identifier}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def respond(self, value, status=200):
        payload = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def check_local_host(self):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            raise ValueError("Open the inspector using its 127.0.0.1 address")

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            self.check_local_host()
            if path == "/api/versions":
                self.respond({"versions": versions()})
            elif path == "/api/jobs":
                with JOBS_LOCK:
                    snapshot = [{k: v for k, v in job.items() if k not in ("cancel", "result")} for job in JOBS.values()]
                self.respond({"jobs": snapshot[-24:][::-1]})
            elif path.startswith("/api/jobs/"):
                with JOBS_LOCK:
                    job = JOBS.get(path.rsplit("/", 1)[1], {})
                    value = job.get("result")
                self.respond(value if value is not None else {"error": "Job result unavailable"},
                             200 if value is not None else 404)
            elif path == "/api/recordings":
                recordings = []
                for file in sorted((STORE / "recordings").glob("*/recording.json"), reverse=True):
                    saved = read_json(file)
                    if saved:
                        recordings.append({key: saved.get(key) for key in ("id", "prompt", "created", "status")})
                self.respond({"recordings": recordings})
            elif path == "/api/benchmarks":
                from urllib.parse import parse_qs

                query = parse_qs(urlparse(self.path).query)
                entries = {v["id"]: v for v in versions()}
                self.respond({"deltas": benchmark_deltas(entries[query["a"][0]], entries[query["b"][0]])})
            elif path.startswith("/api/recordings/"):
                identifier = path.rsplit("/", 1)[1]
                if not all(c.isalnum() or c == "-" for c in identifier):
                    raise ValueError("Invalid recording ID")
                value = read_json(STORE / "recordings" / identifier / "recording.json")
                self.respond(value or {"error": "Recording unavailable"}, 200 if value else 404)
            elif path.startswith("/api/"):
                self.respond({"error": "Unknown endpoint"}, 404)
            elif path in ("/", "/index.html", "/app.js", "/style.css"):
                super().do_GET()
            else:
                self.send_error(404)
        except Exception as error:
            self.respond({"error": str(error)}, 400)

    def do_POST(self):
        try:
            self.check_local_host()
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                raise ValueError("Cross-origin requests are disabled")
            if "application/json" not in self.headers.get("Content-Type", ""):
                raise ValueError("JSON request required")
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 100_000:
                raise ValueError("Request exceeds size limit")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            if self.path == "/api/versions":
                self.respond(submit({**body, "kind": "import"}), 202)
            elif self.path == "/api/jobs":
                self.respond(submit(body), 202)
            elif self.path.startswith("/api/jobs/") and self.path.endswith("/cancel"):
                with JOBS_LOCK:
                    job = JOBS.get(self.path.split("/")[3])
                    if not job or job["status"] not in ACTIVE:
                        raise ValueError("Job is no longer active")
                    job["cancel"].set()
                    job["status"] = "cancelling"
                self.respond({"status": "cancellation requested"})
            else:
                self.respond({"error": "Unknown endpoint"}, 404)
        except Exception as error:
            self.respond({"error": str(error)}, 400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    STORE.mkdir(parents=True, exist_ok=True)
    discover()
    try:
        server = LocalServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        parser.exit(1, f"Cannot bind the inspector to 127.0.0.1:{args.port}. Choose another --port. {error}\n")
    with server:
        print(f"GptCompress Inspector http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            with JOBS_LOCK:
                for job in JOBS.values():
                    job["cancel"].set()
            EXECUTOR.shutdown(wait=True)
