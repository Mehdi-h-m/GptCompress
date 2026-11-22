"""PySide6 desktop shell over the loopback daemon (spec Section 2, Sprint 3).

Three screens — Providers, Data catalog, Run monitor — plus daemon start/stop
and a live connection indicator. Every screen action calls the same daemon
endpoints the `drm` CLI uses, so GUI and CLI stay at parity by construction.

Outcomes are reported in an activity log rather than modal boxes, so flows
are drivable both interactively and programmatically (the offscreen tests).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from drm.cli.plan import _build_payload as build_run_payload
from drm.ui.client import DaemonError, UiClient

TERMINAL_OPERATION_STATES = {"complete", "failed", "cancelled"}


class OperationPoller(QObject):
    """Poll one operation to a terminal state without blocking the UI thread."""

    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, client: UiClient, op_id: str, interval_ms: int = 400, parent=None):
        super().__init__(parent)
        self._client = client
        self._op_id = op_id
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        try:
            doc = self._client.operation(self._op_id)
        except DaemonError as exc:
            self.stop()
            self.failed.emit(f"lost track of {self._op_id}: {exc.message}")
            return
        state = doc["operation"]["state"]
        if state in TERMINAL_OPERATION_STATES:
            self.stop()
            self.finished.emit(doc)


class _Table(QTableWidget):
    """Read-only table helper with row-per-record population."""

    def __init__(self, headers: list[str]):
        super().__init__(0, len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)

    def set_rows(self, rows: list[list[str]]) -> None:
        """Replace rows, preserving selection on the same first-column value.

        Signals stay blocked during the rebuild so dependents (e.g. the copies
        table reacting to artifact selection) see one consistent update instead
        of a cleared intermediate state.
        """
        selected = self.selected_value(0)
        self.blockSignals(True)
        try:
            self.setRowCount(0)
            for row in rows:
                index = self.rowCount()
                self.insertRow(index)
                for column, value in enumerate(row):
                    self.setItem(index, column, QTableWidgetItem(value))
            if selected:
                for row in range(self.rowCount()):
                    if self.item(row, 0).text() == selected:
                        self.selectRow(row)
                        break
        finally:
            self.blockSignals(False)
        if selected:
            self.itemSelectionChanged.emit()

    def selected_value(self, column: int = 0) -> str | None:
        indexes = self.selectionModel().selectedRows()
        if not indexes:
            return None
        item = self.item(indexes[0].row(), column)
        return item.text() if item else None


class ConnectProviderDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect provider")
        self.adapter = QComboBox()
        self.adapter.addItems(["local-runner", "filesystem", "hf", "kaggle", "mock"])
        self.name = QLineEdit()
        self.root = QLineEdit()
        self.repo_id = QLineEdit()
        self.repo_id.setPlaceholderText("owner/drm-store (hf adapter)")
        self.credential_ref = QLineEdit()
        self.credential_ref.setPlaceholderText("env:HF_TOKEN, cred id, or blank for hub default")
        self.region = QLineEdit()
        form = QFormLayout(self)
        form.addRow("Adapter", self.adapter)
        form.addRow("Display name", self.name)
        form.addRow("Store root (filesystem)", self.root)
        form.addRow("Repo id (hf)", self.repo_id)
        form.addRow("Credential ref", self.credential_ref)
        form.addRow("Region", self.region)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def payload(self) -> dict:
        config = {}
        if self.root.text().strip():
            config["root"] = self.root.text().strip()
        if self.repo_id.text().strip():
            config["repo_id"] = self.repo_id.text().strip()
        return {
            "adapter": self.adapter.currentText(),
            "name": self.name.text().strip(),
            "config": config,
            "credential_ref": self.credential_ref.text().strip() or None,
            "region": self.region.text().strip() or None,
        }


class IngestDialog(QDialog):
    def __init__(self, stores: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ingest artifact")
        self.source = QLineEdit()
        self.store = QComboBox()
        for store_row in stores:
            self.store.addItem(f"{store_row['display_name']} ({store_row['id']})", store_row["id"])
        self.kind = QLineEdit("dataset_release")
        self.verify = QCheckBox("Verify stored copy after staging")
        self.verify.setChecked(True)
        form = QFormLayout(self)
        form.addRow("Source file", self.source)
        form.addRow("Store", self.store)
        form.addRow("Kind", self.kind)
        form.addRow(self.verify)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def payload(self) -> dict:
        return {
            "source": self.source.text().strip(),
            "store": self.store.currentData(),
            "kind": self.kind.text().strip() or "dataset_release",
            "verify": self.verify.isChecked(),
        }


class SubmitRunDialog(QDialog):
    def __init__(self, stores: list[dict], task_types: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Submit run")
        self.task = QComboBox()
        self.task.addItems(task_types)
        self.inputs = QLineEdit()
        self.inputs.setPlaceholderText("comma-separated input artifact ids")
        self.params = QLineEdit()
        self.params.setPlaceholderText("k=v pairs separated by commas")
        self.store = QComboBox()
        self.store.addItem("let the planner choose", None)
        for store_row in stores:
            self.store.addItem(f"{store_row['display_name']} ({store_row['id']})", store_row["id"])
        self.name = QLineEdit()
        self.max_attempts = QSpinBox()
        self.max_attempts.setRange(1, 20)
        self.max_attempts.setValue(3)
        form = QFormLayout(self)
        form.addRow("Task type", self.task)
        form.addRow("Inputs", self.inputs)
        form.addRow("Params", self.params)
        form.addRow("Output store", self.store)
        form.addRow("Name", self.name)
        form.addRow("Max attempts", self.max_attempts)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def payload(self) -> dict:
        raw_inputs = [item.strip() for item in self.inputs.text().split(",") if item.strip()]
        raw_params = [item.strip() for item in self.params.text().split(",") if item.strip()]
        return build_run_payload(
            task=self.task.currentText(),
            inputs=raw_inputs or None,
            params=raw_params or None,
            store=self.store.currentData(),
            name=self.name.text().strip() or None,
            experiment=None,
            max_attempts=self.max_attempts.value(),
            timeout=None,
        )


class MainWindow(QMainWindow):
    REFRESH_MS = 2000

    def __init__(self, client: UiClient | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Distributed Resource Manager")
        self.client = client or UiClient()
        self.resize(1080, 640)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setPlaceholderText("Activity log")

        self.connection_label = QLabel("checking daemon...")
        self.connection_label.setAlignment(Qt.AlignRight)

        tabs = QTabWidget()
        tabs.addTab(self._build_providers_tab(), "Providers")
        tabs.addTab(self._build_data_tab(), "Data catalog")
        tabs.addTab(self._build_run_tab(), "Run monitor")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(tabs)
        layout.addWidget(self.log)
        layout.addWidget(self.connection_label)
        self.setCentralWidget(central)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(self.REFRESH_MS)
        self.refresh_timer.timeout.connect(self._on_refresh_tick)
        self.refresh_timer.start()
        self._on_refresh_tick()

    # --- activity log ----------------------------------------------------------

    def log_line(self, message: str) -> None:
        self.log.appendPlainText(message)

    def log_error(self, message: str) -> str:
        line = f"error: {message}"
        self.log.appendPlainText(line)
        return line

    def _guarded(self, label: str, fn, *args, **kwargs):
        """Run a client call, logging failures with their codes; returns the doc or None."""
        try:
            doc = fn(*args, **kwargs)
        except DaemonError as exc:
            self.log_error(f"{label}: {exc.code}: {exc.message}")
            return None
        self.log_line(f"{label}: ok")
        return doc

    # --- daemon control ----------------------------------------------------------

    def start_daemon(self) -> bool:
        from drm.config import ensure_state_dir, log_file, state_dir
        from drm.daemon import control

        try:
            if control.is_healthy(self.client.base_url):
                self.log_line("daemon: already running")
                return True
            ensure_state_dir()
            pid_file = state_dir() / "daemon.pid"
            if pid_file.exists():
                pid_file.unlink()
            pid = control.start_detached(state_dir(), log_file())
            if not control.wait_healthy(self.client.base_url):
                raise DaemonError("daemon_unreachable", f"daemon pid {pid} did not become healthy")
        except (DaemonError, OSError) as exc:
            self.log_error(f"daemon start: {exc}")
            return False
        self.log_line(f"daemon started (pid {pid})")
        self._on_refresh_tick()
        return True

    def stop_daemon(self) -> bool:
        from drm.config import pid_file
        from drm.daemon import control

        try:
            result = control.stop(pid_file(), self.client.base_url)
        except Exception as exc:
            self.log_error(f"daemon stop: {exc}")
            return False
        self.log_line("daemon stopped" if result["was_running"] else "daemon was not running")
        self._on_refresh_tick()
        return True

    def _on_refresh_tick(self) -> None:
        try:
            self.client.health()
            self.connection_label.setText(f"connected  {self.client.base_url}")
            self.connection_label.setStyleSheet("color: #1a7f37;")
        except DaemonError:
            self.connection_label.setText(f"not connected  {self.client.base_url}")
            self.connection_label.setStyleSheet("color: #cf222e;")
        if self.isVisible():
            self.refresh_all()

    def refresh_all(self) -> None:
        self.refresh_providers()
        self.refresh_data()
        self.refresh_runs()

    # --- providers tab -------------------------------------------------------------

    def _build_providers_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.providers_table = _Table(["id", "adapter", "name", "kind", "state", "region"])
        layout.addWidget(self.providers_table)
        self.provider_detail = QPlainTextEdit()
        self.provider_detail.setReadOnly(True)
        self.provider_detail.setMaximumHeight(110)
        layout.addWidget(self.provider_detail)
        buttons = QHBoxLayout()
        for text, handler in (
            ("Connect...", self._on_connect_provider),
            ("Probe", self._on_probe_provider),
            ("Pause", self._on_pause_provider),
            ("Resume", self._on_resume_provider),
            ("Refresh", self.refresh_providers),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def refresh_providers(self) -> None:
        try:
            providers = self.client.providers()
        except DaemonError as exc:
            self.providers_table.set_rows([])
            self.log_error(f"providers refresh: {exc.message}")
            return
        self.providers_table.set_rows(
            [
                [p["id"], p["adapter"], p["display_name"], p["kind"], p["state"], p.get("region") or "-"]
                for p in providers
            ]
        )

    def _selected_provider(self) -> str | None:
        return self.providers_table.selected_value(0)

    def _on_connect_provider(self) -> None:
        dialog = ConnectProviderDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        doc = self._guarded("provider connect", self.client.connect_provider, dialog.payload())
        if doc:
            self.log_line(f"connected provider {doc['provider']['id']}")
        self.refresh_providers()

    def connect_provider(self, payload: dict) -> dict:
        """Programmatic path used by tests and dialogs."""
        doc = self.client.connect_provider(payload)
        self.log_line(f"connected provider {doc['provider']['id']}")
        self.refresh_providers()
        return doc

    def _on_probe_provider(self) -> None:
        provider_id = self._selected_provider()
        if not provider_id:
            self.log_error("probe: select a provider first")
            return
        doc = self._guarded(f"probe {provider_id}", self.client.probe_provider, provider_id)
        if doc:
            import json

            self.provider_detail.setPlainText(json.dumps(doc["probe"], indent=2, sort_keys=True))

    def _on_pause_provider(self) -> None:
        provider_id = self._selected_provider()
        if provider_id:
            self._guarded(f"pause {provider_id}", self.client.set_provider_state, provider_id, "pause")
        self.refresh_providers()

    def _on_resume_provider(self) -> None:
        provider_id = self._selected_provider()
        if provider_id:
            self._guarded(f"resume {provider_id}", self.client.set_provider_state, provider_id, "resume")
        self.refresh_providers()

    # --- data tab ------------------------------------------------------------------

    def _build_data_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.artifacts_table = _Table(["id", "kind", "bytes", "sha256", "pinned"])
        layout.addWidget(self.artifacts_table)
        self.copies_table = _Table(["copy id", "provider", "status", "locator"])
        layout.addWidget(self.copies_table)
        buttons = QHBoxLayout()
        for text, handler in (
            ("Ingest...", self._on_ingest),
            ("Verify copy", self._on_verify_copy),
            ("Pin artifact", self._on_pin_artifact),
            ("Evict copy", self._on_evict_copy),
            ("Refresh", self.refresh_data),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.artifacts_table.itemSelectionChanged.connect(self._on_artifact_selected)
        return widget

    def refresh_data(self) -> None:
        try:
            artifacts = self.client.artifacts()
        except DaemonError as exc:
            self.artifacts_table.set_rows([])
            self.log_error(f"data refresh: {exc.message}")
            return
        self._artifacts_by_id = {a["id"]: a for a in artifacts}
        self.artifacts_table.set_rows(
            [
                [
                    a["id"],
                    a["kind"],
                    str(a["byte_count"] or 0),
                    a["content_hash"][:12],
                    "yes" if a.get("pinned") else "",
                ]
                for a in artifacts
            ]
        )

    def _on_artifact_selected(self) -> None:
        artifact_id = self.artifacts_table.selected_value(0)
        artifact = self._artifacts_by_id.get(artifact_id) if artifact_id else None
        if not artifact:
            self.copies_table.set_rows([])
            return
        self.copies_table.set_rows(
            [
                [c["id"], c["provider_id"], c["status"], c["locator"]]
                for c in artifact.get("copies", [])
            ]
        )

    def _selected_copy_id(self) -> str | None:
        return self.copies_table.selected_value(0)

    def ingest_artifact(self, payload: dict) -> str:
        """Programmatic ingest; returns the operation id and polls to terminal."""
        doc = self.client.ingest(payload)
        op_id = doc["operation"]["id"]
        self.log_line(f"ingest queued as {op_id}")
        poller = OperationPoller(self.client, op_id, parent=self)
        poller.finished.connect(self._on_ingest_finished)
        poller.failed.connect(self.log_error)
        poller.start()
        return op_id

    def _on_ingest_finished(self, doc: dict) -> None:
        operation = doc["operation"]
        if operation["state"] == "complete":
            artifact = operation["result"]["artifact"]
            self.log_line(f"ingest complete: artifact {artifact['id']} ({artifact['byte_count']} B)")
        else:
            error = operation.get("error") or {}
            self.log_error(f"ingest failed: {error.get('code')}: {error.get('message')}")
        self.refresh_data()

    def _on_ingest(self) -> None:
        try:
            stores = [p for p in self.client.providers() if p["kind"] in ("storage", "combined")]
        except DaemonError as exc:
            self.log_error(f"ingest: {exc.message}")
            return
        dialog = IngestDialog(stores, self)
        if dialog.exec() != QDialog.Accepted:
            return
        self.ingest_artifact(dialog.payload())

    def _on_verify_copy(self) -> None:
        copy_id = self._selected_copy_id()
        if not copy_id:
            self.log_error("verify: select a copy first")
            return
        doc = self._guarded(f"verify {copy_id}", self.client.verify, copy_id)
        if doc:
            result = doc["operation"]["result"]
            self.log_line(f"verified {copy_id} (sha256:{result['checksum'][:12]})")

    def _on_pin_artifact(self) -> None:
        artifact_id = self.artifacts_table.selected_value(0)
        if not artifact_id:
            self.log_error("pin: select an artifact first")
            return
        self._guarded(f"pin {artifact_id}", self.client.pin, artifact_id)
        self.refresh_data()

    def _on_evict_copy(self) -> None:
        copy_id = self._selected_copy_id()
        if not copy_id:
            self.log_error("evict: select a copy first")
            return
        self._guarded(f"evict {copy_id}", self.client.evict, copy_id)
        self.refresh_data()

    # --- run monitor tab ------------------------------------------------------------

    def _build_run_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.jobs_table = _Table(["job id", "name", "state", "updated"])
        layout.addWidget(self.jobs_table)
        self.run_detail = QPlainTextEdit()
        self.run_detail.setReadOnly(True)
        self.run_detail.setMaximumHeight(150)
        layout.addWidget(self.run_detail)
        buttons = QHBoxLayout()
        for text, handler in (
            ("Preview / Submit...", self._on_submit_run),
            ("Cancel", self._on_cancel_job),
            ("Retry", self._on_retry_job),
            ("Refresh", self.refresh_runs),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.jobs_table.itemSelectionChanged.connect(self._on_job_selected)
        return widget

    def refresh_runs(self) -> None:
        try:
            providers = self.client.providers()
            jobs = self.client.jobs()
        except DaemonError as exc:
            self.jobs_table.set_rows([])
            self.log_error(f"run refresh: {exc.message}")
            return
        self._storage_providers = [p for p in providers if p["kind"] in ("storage", "combined")]
        self.jobs_table.set_rows(
            [
                [j["id"], j.get("name") or "-", j["state"], j["updated_at"]]
                for j in jobs
            ]
        )

    def preview_plan(self, payload: dict) -> dict:
        """Programmatic preview; shows blockers and the decision reference."""
        doc = self.client.preview_plan(payload)
        plan_doc = doc["plan"]
        if plan_doc["blockers"]:
            self.log_error("plan blockers: " + "; ".join(plan_doc["blockers"]))
        else:
            chosen = plan_doc["chosen"]
            self.log_line(
                f"plan {plan_doc['decision_id']}: compute {chosen['compute']},"
                f" store {chosen['output_store']}"
            )
        return plan_doc

    def submit_run(self, payload: dict) -> str:
        """Programmatic submit; returns the run operation id and polls it."""
        doc = self.client.submit_run(payload)
        op_id = doc["operation"]["id"]
        self.log_line(
            f"job {doc['job']['id']} submitted ({doc['job']['state']}); operation {op_id}"
        )
        self._submitted_jobs = getattr(self, "_submitted_jobs", [])
        self._submitted_jobs.append({"job_id": doc["job"]["id"], "op_id": op_id})
        poller = OperationPoller(self.client, op_id, parent=self)
        poller.finished.connect(self._on_run_finished)
        poller.failed.connect(self.log_error)
        poller.start()
        self.refresh_runs()
        return op_id

    def _on_run_finished(self, doc: dict) -> None:
        operation = doc["operation"]
        result = operation.get("result") or {}
        if operation["state"] == "complete":
            for item in result.get("published", []):
                artifact = item["artifact"]
                self.log_line(
                    f"run complete: published {artifact['id']} ({artifact['kind']})"
                )
        else:
            error = operation.get("error") or {}
            self.log_error(f"run ended: {operation['state']} ({error.get('code')})")
        self.refresh_runs()

    def _on_submit_run(self) -> None:
        dialog = SubmitRunDialog(self._storage_providers, sorted(_task_types()), self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            payload = dialog.payload()
        except Exception as exc:
            self.log_error(f"submit form: {exc}")
            return
        try:
            self.submit_run(payload)
        except DaemonError as exc:
            self.log_error(f"submit: {exc.message}")

    def _selected_job_id(self) -> str | None:
        return self.jobs_table.selected_value(0)

    def _on_job_selected(self) -> None:
        job_id = self._selected_job_id()
        if not job_id:
            return
        try:
            doc = self.client.run(job_id)
        except DaemonError:
            return
        import json

        self.run_detail.setPlainText(json.dumps(doc, indent=2, default=str, sort_keys=True))

    def _on_cancel_job(self) -> None:
        job_id = self._selected_job_id()
        if not job_id:
            self.log_error("cancel: select a job first")
            return
        self._guarded(f"cancel {job_id}", self.client.cancel_run, job_id)
        self.refresh_runs()

    def _on_retry_job(self) -> None:
        job_id = self._selected_job_id()
        if not job_id:
            self.log_error("retry: select a job first")
            return
        self._guarded(f"retry {job_id}", self.client.retry_job, job_id)
        self.refresh_runs()


def _task_types() -> list[str]:
    from drm.workers.tasks import TASK_REGISTRY

    return list(TASK_REGISTRY)
