"""Face Scan package.

This package owns the dedicated Face Scan service layer so the UI and API can
share one contract instead of formatting ad-hoc face results in multiple places.
"""

from basetruth.face_scan.service import run_face_scan_static

__all__ = ["run_face_scan_static"]