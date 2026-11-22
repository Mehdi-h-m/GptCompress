"""Entry point for the desktop shell: ``drm ui`` or ``drm-ui``."""

from __future__ import annotations


def main() -> None:
    from PySide6.QtWidgets import QApplication

    from drm.ui.app import MainWindow

    app = QApplication(["drm"])
    app.setApplicationName("Distributed Resource Manager")
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
