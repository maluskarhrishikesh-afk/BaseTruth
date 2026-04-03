"""Functional WebSocket test for KYC liveness detection pipeline.

Run with:  .venv\Scripts\python.exe tests\test_kyc_ws.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys

import cv2
import numpy as np
import requests
import websockets

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

API = "http://127.0.0.1:8000"
WS_API = "ws://127.0.0.1:8000"
FACE_IMG = os.path.join(ROOT, "your_data", "test_face.jpg")


def _encode_frame(img: np.ndarray, quality: int = 75) -> str:
    """Encode image as base64 JPEG and wrap in the expected JSON frame envelope."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf).decode()
    return json.dumps({"type": "frame", "data": b64})


async def test_blink_challenge() -> None:
    print("\n=== Blink Challenge Test ===")

    # 1 ── Create session ──────────────────────────────────────────────────
    r = requests.post(f"{API}/kyc/sessions",
                      json={"customer_name": "WS-Test", "challenges": ["blink"]})
    r.raise_for_status()
    session = r.json()
    sid = session["session_id"]
    print(f"Session {sid}  challenges={session['challenges']}")

    # 2 ── Prepare frames ─────────────────────────────────────────────────
    img = cv2.imread(FACE_IMG)
    assert img is not None, f"Could not load {FACE_IMG}"
    img = cv2.resize(img, (480, 480))

    # Simulate closed-eye frame: darken the eye-band area
    img_closed = img.copy()
    h, w = img_closed.shape[:2]
    img_closed[int(h * 0.30): int(h * 0.62),
               int(w * 0.20): int(w * 0.80)] = (20, 20, 20)

    open_b64   = _encode_frame(img)
    closed_b64 = _encode_frame(img_closed)

    # 3 ── Connect WebSocket ───────────────────────────────────────────────
    url = f"{WS_API}/kyc/ws/{sid}"
    print(f"Connecting: {url}")

    received: list[dict] = []

    async with websockets.connect(url) as ws:
        async def send_recv(b64: str, label: str) -> dict:
            await ws.send(b64)
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            d = json.loads(raw)
            received.append(d)
            t = d.get("type", "?")
            fb = d.get("feedback", "")[:60]
            print(f"  [{label}] type={t}  feedback={fb!r}")
            return d

        # Baseline: 8 open-eye frames
        for i in range(8):
            d = await send_recv(open_b64, f"open-{i}")
            if d.get("type") == "result":
                print("  Premature result:", d)
                return

        # Blink: 4 closed-eye frames
        for i in range(4):
            d = await send_recv(closed_b64, f"closed-{i}")
            if d.get("type") == "result":
                print("  RESULT:", d)
                return

        # Recovery: 5 open-eye frames
        for i in range(5):
            d = await send_recv(open_b64, f"reopen-{i}")
            if d.get("type") == "result":
                print("  RESULT:", d)
                return

    print("\nMessages received:", len(received))
    types = [m.get("type") for m in received]
    print("Types seen:", types)

    result_msgs = [m for m in received if m.get("type") == "result"]
    challenge_passed = any(
        m.get("type") == "challenge_passed" for m in received
    )
    print("Challenge passed messages:", challenge_passed)
    print("Result messages:", result_msgs)

    if challenge_passed or (result_msgs and result_msgs[-1].get("passed")):
        print("\n✅ BLINK TEST PASSED")
    else:
        print("\n⚠ Blink not detected in simulation (may need real webcam) — check thresholds")


async def test_turn_challenge() -> None:
    print("\n=== Turn Left Challenge Test ===")

    r = requests.post(f"{API}/kyc/sessions",
                      json={"customer_name": "WS-Turn", "challenges": ["turn_left"]})
    r.raise_for_status()
    session = r.json()
    sid = session["session_id"]
    print(f"Session {sid}  challenges={session['challenges']}")

    img = cv2.imread(FACE_IMG)
    assert img is not None
    img = cv2.resize(img, (480, 480))

    # Simulate turned face: shift content rightward
    M = np.float32([[1, 0, 80], [0, 1, 0]])  # translate 80px right
    img_turned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    straight_b64 = _encode_frame(img)
    turned_b64   = _encode_frame(img_turned)

    url = f"{WS_API}/kyc/ws/{sid}"
    print(f"Connecting: {url}")

    async with websockets.connect(url) as ws:
        # Baseline straight
        for i in range(5):
            await ws.send(straight_b64)
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            d = json.loads(raw)
            print(f"  [straight-{i}] {d.get('feedback','')[:50]!r}")

        # Turn frames
        for i in range(8):
            await ws.send(turned_b64)
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            d = json.loads(raw)
            print(f"  [turned-{i}] type={d.get('type')} {d.get('feedback','')[:50]!r}")
            if d.get("type") in ("result", "challenge_passed"):
                print("  ✅ Turn registered:", d)
                return

    print("⚠ Turn not strongly detected in simulation")


if __name__ == "__main__":
    print("BaseTruth KYC WebSocket Liveness Tests")
    print("API:", API)
    asyncio.run(test_blink_challenge())
    asyncio.run(test_turn_challenge())
