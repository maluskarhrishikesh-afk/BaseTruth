"""ML Training Pipeline page — train the fraud-detection models from the UI.

This page lets you:
  1. See the current status of both the Image and PDF fraud models.
  2. Choose which model(s) to retrain.
  3. Watch live training progress streaming over a WebSocket connection.
  4. Explore the results through plain-English charts that explain:
       - How your training data is split across the 4 verdict classes.
       - Which forensic signals had the most influence on each verdict.
       - A confusion matrix showing where the model is most/least confident.
  5. Read a plain-English guide to all 11 forensic signals the system uses.
"""
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from basetruth.logger import get_logger
from basetruth.ui.components import _page_title

log = get_logger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
_REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent.parent
_IMAGE_CSV   = _REPO_ROOT / "fraud_model" / "data"   / "training_data_image.csv"
_PDF_CSV     = _REPO_ROOT / "fraud_model" / "data"   / "training_data_pdf.csv"
_LIVE_CSV    = _REPO_ROOT / "fraud_model" / "data"   / "training_data_face_scan_live.csv"
_IMAGE_PKL   = _REPO_ROOT / "fraud_model" / "models" / "ml_scorer_image.pkl"
_PDF_PKL     = _REPO_ROOT / "fraud_model" / "models" / "ml_scorer_pdf.pkl"
_LIVE_PKL    = _REPO_ROOT / "fraud_model" / "models" / "ml_scorer_face_scan_live.pkl"

# ─── API server connection ─────────────────────────────────────────────────────
_API_PORT = 8000


# ─── Class colours — binary palette ──────────────────────────────────────────
_CLASS_COLORS  = ["#22c55e", "#ef4444"]
_CLASS_LABELS  = ["ORIGINAL", "FAKE/EDITED"]

# Face Scan Live binary labels (used by the live model card and charts).
_LIVE_CLASS_LABELS = ["GENUINE", "SPOOF"]

# ─── Plain-English guide to the image forensic signals ───────────────────────
# Each entry: (signal_name, emoji, one_line_summary, longer_explanation)
_SIGNAL_GUIDE = [
    (
        "ela_mean",
        "🔴",
        "ELA mean — average freshness difference across all pixels",
        "When you save a JPEG, every pixel gets slightly softened by compression. "
        "ELA re-saves the image at a lower quality and measures how much each pixel changed. "
        "'Mean' is the average change across the whole image. A higher average often indicates "
        "overall re-editing — the document was opened, changed, and re-saved at least once.",
    ),
    (
        "ela_max",
        "🟥",
        "ELA max — single hottest pixel in the freshness map",
        "Even if the average ELA is low, a single extremely hot pixel can reveal a small but "
        "targeted edit. A forger who changes just one digit in a salary field will leave a tiny "
        "bright spot in the ELA map. 'Max' captures that worst-case spike that the mean "
        "might dilute.",
    ),
    (
        "ela_std",
        "📉",
        "ELA std — how unevenly fresh the compression is across the image",
        "In a genuine document, every region was compressed at the same time, so ELA values "
        "are uniformly low everywhere. A tampered document has patched regions that were "
        "compressed separately — causing some tiles to be much hotter than others. "
        "High standard deviation = uneven compression history = likely tampering.",
    ),
    (
        "ela_suspicious_block_ratio",
        "🧩",
        "ELA suspicious block ratio — fraction of tiles above the suspicion threshold",
        "The image is tiled into 32×32 pixel blocks. Any block whose ELA value is more than "
        "2.5× the image average is counted as suspicious. This ratio is how many blocks "
        "triggered that threshold. Values above 0.05 (5% of the image) are a strong sign "
        "of patching or copy-paste.",
    ),
    (
        "metadata_flag_count",
        "🏷️",
        "Metadata flags — count of contradictory hidden file details",
        "Every image and PDF carries hidden metadata: creation date, camera model, software "
        "used, GPS location, etc. When a document is forged, fraudsters often forget to clean "
        "up this metadata, or they accidentally introduce contradictions — for example, the "
        "creation date is 2024 but the camera model is from 2030. This count is how many "
        "such suspicious or contradictory metadata flags were found.",
    ),
    (
        "file_entropy_bits",
        "🎲",
        "File entropy — how 'random' the raw file bytes are (bits per byte, 0–8)",
        "Genuine compressed files (JPEG, PDF) are highly random-looking because compression "
        "removes repetitive patterns. The entropy of a real document is typically 7.0–8.0 "
        "bits/byte. A value significantly below 7.0 suggests the file was tampered with in "
        "a way that introduced repetitive regions, or that it is an uncompressed format "
        "masquerading as a compressed one. (Note: Currently used by the Heuristic engine; "
        "excluded from the ML image model due to zero variance in the training dataset.)",
    ),
    (
        "noise_hotspot_ratio",
        "📡",
        "Noise hotspot ratio — fraction of tiles with outlier noise levels",
        "Real camera photos and scanned documents have a fine, even grain of noise distributed "
        "uniformly across the image. When a region has been replaced digitally, that patched "
        "region is often suspiciously smooth compared to the rest of the document. This ratio "
        "counts tiles whose noise level is drastically different from the image median.",
    ),
    (
        "dct_comb_ratio",
        "📊",
        "DCT comb ratio — strength of the double-compression pattern",
        "JPEG uses maths called DCT to compress images. When a document is tampered (edited "
        "in Photoshop, then re-saved), DCT compression runs twice. The second pass leaves a "
        "distinctive 'comb' pattern in the frequency data. A ratio above 1.3 strongly suggests "
        "the file has been through at least two save cycles — exactly what happens when someone "
        "edits and re-exports.",
    ),
    (
        "dct_skipped",
        "⏭️",
        "DCT skipped — 1 if this is a PNG/non-JPEG file, 0 if JPEG",
        "DCT analysis only applies to JPEG files. When the file is PNG or another lossless "
        "format, DCT is skipped and this flag is set to 1. The model can use this as a "
        "format indicator — for example, original photographs are almost always JPEG while "
        "digitally-created fakes are sometimes exported as PNG to avoid JPEG artefacts.",
    ),
    (
        "clone_ratio",
        "🔁",
        "Clone ratio — fraction of keypoints that are copy-pasted within the image",
        "A common fraud technique is to copy a region from one part of a document (e.g., an "
        "amount field) and paste it over another region (e.g., to change a salary figure). "
        "Clone detection scans the whole image for blocks of pixels that look identical to "
        "other blocks. High values indicate internal copy-paste, which is almost always fraudulent.",
    ),
    (
        "color_anomaly_ratio",
        "🎨",
        "Colour anomaly ratio — fraction of pixels outside the document's natural palette",
        "Photo editing tools often leave blobs of flat, unnaturally uniform colour when "
        "content is erased or painted over. This fraction measures how many pixels fall "
        "outside the statistical colour palette of the surrounding document. High values "
        "suggest large areas of artificially uniform colour — a classic sign of retouching.",
    ),
    (
        "color_largest_blob_px",
        "🖼️",
        "Largest colour blob — area in pixels of the biggest anomalous colour region",
        "Even if the overall colour anomaly ratio is small, one very large blob of uniform "
        "colour is a strong indicator. Think of the white rectangle you see when someone "
        "digitally covers a field before printing and scanning. This value captures the size "
        "of the worst single anomalous region rather than the average.",
    ),
    (
        "edge_high_density_ratio",
        "📐",
        "Edge density ratio — fraction of tiles with unnaturally sharp or cluttered edges",
        "Genuine printed-and-scanned documents have a characteristic softness from the "
        "printing and scanning process. When text or graphics are digitally inserted, they "
        "have perfectly sharp edges that stand out as 'too clean'. This ratio counts tiles "
        "where the edge density is statistically higher than natural document regions.",
    ),
    (
        "saturation_ratio",
        "🌈",
        "Saturation ratio — fraction of tiles with over-saturated colours",
        "Genuine scanned documents have muted, slightly faded colours from the scan process. "
        "Digitally-generated or digitally-edited regions are often more vibrant because they "
        "came directly from a screen render. This ratio measures how many tiles have "
        "saturation levels above what is expected for a scanned physical document.",
    ),
    (
        "font_stroke_cv",
        "🔤",
        "Font stroke CV — coefficient of variation of character stroke widths",
        "Every genuine document was typeset in one or two fonts with consistent stroke widths. "
        "If someone replaces one digit using a different font or screen-capture tool, the stroke "
        "thickness of that digit will differ from its neighbours. CV (standard deviation ÷ mean) "
        "measures this inconsistency across the whole document. High CV = likely font mixing.",
    ),
    (
        "font_suspicious_regions",
        "🔢",
        "Font suspicious regions — count of spatially-clustered font anomaly groups",
        "Rather than flagging individual characters, this groups nearby font anomalies into "
        "'regions'. A single suspicious region could mean one field was edited. Multiple "
        "suspicious regions spread across the document suggests a more extensive alteration. "
        "Each region is a cluster of characters whose stroke or shape statistics differ "
        "significantly from the document's established font baseline.",
    ),
    (
        "font_sharpness_outlier_ratio",
        "🔍",
        "Font sharpness outlier ratio — fraction of character regions with outlier focus",
        "When a number or word is digitally typed or pasted into a scanned document, it is "
        "typically sharper than the surrounding text (which has the natural softness of a "
        "physical scan). This ratio measures how many character bounding boxes have a "
        "sharpness score that is a statistical outlier compared to the rest of the page. "
        "A high ratio indicates that some text was digitally inserted.",
    ),
    (
        "font_skipped",
        "⏭️",
        "Font skipped — 1 if no text was found or font analysis was skipped",
        "Font analysis requires at least some readable text regions. For documents that are "
        "purely graphical (e.g. a photograph with no text), the font layer is skipped. "
        "The model can learn that font_skipped=1 shifts which other signals matter most "
        "for that type of document.",
    ),
    (
        "ai_spike_ratio",
        "🤖",
        "AI spike ratio — strength of the AI/GAN upsampling grid pattern in the FFT",
        "AI-generated images (from tools like Midjourney, DALL-E, or Stable Diffusion) often "
        "leave a characteristic repeating pattern in the frequency spectrum. This signal "
        "analyses the Fourier transform and measures how tall the periodic spikes are relative "
        "to the noise floor. Ratios above 3.5 indicate strong AI generation or AI upsampling.",
    ),
]

# ─── Plain-English guide to the PDF forensic signals ─────────────────────────
# Each entry: (signal_name, emoji, one_line_summary, longer_explanation)
# These 17 signals match PDF_FEATURE_NAMES in basetruth.analysis.ml_scorer_pdf exactly.
# The CSV has more columns, but many are dropped before training (leakage, constants, etc.).
_PDF_SIGNAL_GUIDE = [
    (
        "incremental_updates",
        "🔄",
        "Incremental updates — how many times the PDF was re-saved after creation",
        "PDF allows 'incremental updates': rather than rewriting the whole file, changes are "
        "appended to the end. A genuine payroll or HR document is created once and never "
        "re-saved. Each incremental update is a sign the file was opened and altered. "
        "A count above 0 is a strong red flag for post-creation tampering.",
    ),
    (
        "eof_marker_count",
        "🚩",
        "EOF marker count — number of %%EOF end-of-file markers",
        "A valid PDF has exactly one %%EOF marker at the very end of the file. Each "
        "incremental update adds another %%EOF. If this count is greater than 1, the file "
        "has been modified at least once after initial creation — something that should "
        "never happen for a genuine institutional document.",
    ),
    (
        "metadata_anomaly_score",
        "🏷️",
        "Metadata anomaly score — 0–100 composite score of suspicious metadata",
        "A single numeric score that combines metadata flags, creation/modification date "
        "gaps, and creator-software anomalies. A score of 0 means every metadata field "
        "is consistent and credible. Each inconsistency (e.g. modification date before "
        "creation date, creator is a screen-capture tool) pushes the score higher. "
        "Scores above 30 are considered suspicious.",
    ),
    (
        "hidden_text_spans",
        "👻",
        "Hidden text spans — total white or zero-size text runs in the document",
        "Fraudsters sometimes overlay invisible text on top of visible text — for example "
        "a white-coloured '5' placed precisely over a genuine '3' to change a salary figure. "
        "This is nearly impossible to spot by eye but trivially detectable in the raw PDF "
        "stream. Any non-zero count is a very strong tampering indicator.",
    ),
    (
        "white_text_spans",
        "⬜",
        "White text spans — text rendered in white ink, invisible against white paper",
        "A focused subset of hidden_text_spans: text whose colour is explicitly white "
        "or near-white. This is the classic overwrite technique — a white block covers "
        "the original value; the fake value is printed on top. Even a single white-text "
        "span in a payroll document warrants immediate investigation.",
    ),
    (
        "javascript_count",
        "⚙️",
        "JavaScript count — number of JavaScript actions embedded in the PDF",
        "JavaScript inside a PDF can silently modify content, call remote servers, or "
        "trigger further actions on open. No genuine payroll or employment document ever "
        "contains JavaScript. Any value above 0 is immediately suspicious and in some "
        "cases indicates a weaponised or fully synthetic document.",
    ),
    (
        "embedded_files_count",
        "📎",
        "Embedded files — count of files attached inside the PDF",
        "PDFs can carry hidden file attachments. Genuine HR documents contain no "
        "attachments. The presence of embedded files suggests the document has been "
        "modified to include hidden data — inconsistent with a legitimate payslip or "
        "offer letter generated directly from HR software.",
    ),
    (
        "signature_gap_score",
        "🕳️",
        "Signature gap score — 0–100 measure of unsigned bytes after the last signature",
        "A digital signature 'covers' the exact bytes of the file at signing time. If the "
        "file is modified afterwards, new content appears outside the signed byte range. "
        "This score quantifies how large those gaps are. Any positive value means content "
        "was added or changed after signing — definitive proof of post-signature modification.",
    ),
    (
        "render_ela_suspicious_block_ratio",
        "🧩",
        "Render ELA suspicious blocks — fraction of rendered page tiles above the suspicion threshold",
        "The page is rendered to a bitmap and divided into 32×32 pixel tiles. Any tile "
        "whose Error Level Analysis (ELA) value exceeds 2.5× the page average is flagged. "
        "This ratio measures what fraction of the page is 'suspiciously fresh' — indicating "
        "areas that were digitally altered and re-compressed after the original was created.",
    ),
    (
        "render_noise_hotspot_ratio",
        "📡",
        "Render noise hotspot ratio — fraction of rendered page tiles with anomalous noise",
        "Genuine pages rendered from a single-origin PDF have a consistent noise texture. "
        "Pasted or replaced regions are often smoother or noisier than surrounding content. "
        "This signal counts tiles whose noise level is a statistical outlier from the page "
        "median, revealing digitally-inserted regions on the rendered page.",
    ),
    (
        "object_count",
        "🗂️",
        "Object count — total PDF objects in the cross-reference table",
        "Every element of a PDF (page, font, image, text block) is a numbered object. "
        "A genuine single-page payslip typically contains 30–120 objects. Abnormally high "
        "counts suggest hidden content, unused 'ghost' objects left behind by an editor, "
        "or complex object-stream structures used to obscure changes.",
    ),
    (
        "stream_entropy",
        "🎲",
        "Stream entropy — Shannon entropy of the raw file bytes (0–8 bits/byte)",
        "A well-compressed genuine PDF has entropy between 7.0 and 8.0 bits per byte "
        "because compression removes repetition. A value below 7.0 can indicate the file "
        "was reconstructed, inflated with padding, or had compressed streams replaced with "
        "uncompressed content — all consistent with editing tools that rebuild the file "
        "structure when saving.",
    ),
    (
        "xref_mismatch_score",
        "❌",
        "XRef mismatch score — 0–100 score for cross-reference table integrity issues",
        "The PDF cross-reference table (XRef) maps every object to its byte offset. When "
        "a file is tampered with, the byte offsets in the XRef often no longer match the "
        "actual locations of objects. A high mismatch score means the XRef has been "
        "partially or incorrectly regenerated — a hallmark of round-trip editing.",
    ),
    (
        "font_switch_score",
        "🔤",
        "Font switch score — 0–100 proxy for font mixing across text regions",
        "A genuine payslip or offer letter is typeset in one consistent font family. "
        "If someone replaces a number using a different application, the replacement text "
        "uses a different font. This score quantifies how much font diversity is present "
        "relative to what is expected. High scores indicate multi-source text assembly — "
        "a strong sign of field-level tampering.",
    ),
    (
        "ocr_text_layer_gap",
        "🔍",
        "OCR text-layer gap — 0–100 measure of divergence between OCR and embedded text",
        "A scanned-and-OCR'd PDF has two text layers: the visible image and the embedded "
        "text added by OCR. For a genuine document, these two layers agree closely. If "
        "the embedded text has been altered without updating the image (or vice versa), "
        "the gap between what is visible and what is searchable is anomalously large — "
        "exactly what happens when a value is changed in the text layer only.",
    ),
    (
        "is_scanned_pdf",
        "📸",
        "Is scanned PDF — 1 if the document appears to be a physical scan",
        "Documents that originated as physical paper scans have different characteristics "
        "from digitally-generated PDFs. Scanned documents are more likely to be tampered "
        "by image editing (rather than PDF editing), so the model weights other signals "
        "differently when this flag is 1. It is also useful on its own — claiming to be "
        "a native digital document while actually being a scan is a red flag.",
    ),
    (
        "has_signature",
        "✍️",
        "Has signature — 1 if any digital signature field exists in the PDF",
        "Some genuine documents carry an official digital signature from the issuing "
        "organisation. If the content has been modified after signing, the signature "
        "becomes cryptographically invalid — but the field itself remains. Combined with "
        "signature_gap_score, this detects the signed-then-tampered scenario where a "
        "legitimate signature is present but no longer covers the full document content.",
    ),
]

# ─── Plain-English guide to the Face Scan Live signals ───────────────────────
# Each entry: (signal_name, emoji, one_line_summary, longer_explanation)
# These 20 signals match FEATURE_NAMES in basetruth.face_scan.ml_scorer_live exactly.
_LIVE_SIGNAL_GUIDE = [
    (
        "yaw_jerk",
        "↔️",
        "Yaw jerk — how abruptly the head turned left or right between frames",
        "A real person turns their head smoothly and continuously. A screen recording "
        "or replay attack plays back at a fixed frame rate, often with quantised motion that "
        "jumps abruptly between frames. Yaw jerk is the mean absolute second derivative of "
        "the left-right head-angle signal. High values indicate non-smooth motion — suspicious.",
    ),
    (
        "pitch_jerk",
        "↕️",
        "Pitch jerk — how abruptly the head nodded up or down between frames",
        "Analogous to yaw_jerk but for the up-down nodding axis. Real nods have smooth "
        "acceleration and deceleration curves. Replayed or synthesised video often has "
        "step-function jumps in pitch because the source video was edited or speed-ramped. "
        "High pitch_jerk combined with high yaw_jerk strongly indicates non-live content.",
    ),
    (
        "nose_jitter",
        "👃",
        "Nose jitter — frame-to-frame variability of the nose tip landmark position",
        "InsightFace localises the nose tip in every frame. A real person sitting still has "
        "sub-pixel micro-tremor from breathing and muscle activity. A static photo held in "
        "front of the camera has near-zero jitter. A screen recording has only encoding "
        "noise. Unusually low jitter is a spoofing indicator.",
    ),
    (
        "temporal_consistency_score",
        "📈",
        "Temporal consistency score — overall smoothness of head motion across all frames (0–100)",
        "A composite 0–100 score (higher = more suspicious) that combines yaw jerk, pitch "
        "jerk, and nose jitter into a single heuristic. Computed by the temporal consistency "
        "check inside the Face Scan Live pipeline. The ML model learns when this heuristic "
        "is well-calibrated and when it needs correction.",
    ),
    (
        "repeat_frame_score",
        "🔁",
        "Repeat frame score — fraction of consecutive frame pairs that are nearly identical (0–100)",
        "Replay attacks often loop a short video clip or pause on a single frame. This score "
        "measures how often consecutive frames have a very high perceptual hash similarity. "
        "A genuine live session always has at least small changes between every frame pair "
        "due to breathing, blinking, and micro-saccades.",
    ),
    (
        "flicker_score",
        "💡",
        "Flicker score — strength of periodic brightness oscillation across frames (0–100)",
        "Camera-facing-a-screen attacks introduce a characteristic brightness flicker caused "
        "by the mismatch between the screen's refresh rate and the camera's shutter speed. "
        "This score measures the amplitude of the periodic brightness component in the "
        "brightness time-series. High flicker = possible screen.",
    ),
    (
        "brightness_instability",
        "🌓",
        "Brightness instability — non-periodic variance in frame brightness",
        "While flicker_score captures periodic oscillations, brightness_instability captures "
        "random brightness jumps. This is useful for detecting low-quality replay attacks "
        "where the source video had poor exposure control, or where the attacker is physically "
        "moving the replay device during the session.",
    ),
    (
        "mean_eye_jitter",
        "👁️",
        "Mean eye jitter — average involuntary micro-movement of both eyes across frames",
        "Human eyes are never perfectly still — involuntary micro-saccades cause tiny "
        "movements even when staring straight ahead. A photo or a screen recording has "
        "zero eye jitter because the eye pixels are static. This is one of the strongest "
        "single-frame liveness signals: near-zero eye jitter almost always indicates a "
        "non-live source.",
    ),
    (
        "iod_yaw_correlation",
        "📏",
        "IOD-yaw correlation — relationship between head rotation and inter-ocular distance",
        "When a real 3D head turns sideways, one eye gets closer to the camera and the "
        "other gets further away — so the inter-ocular distance (IOD, the pixel distance "
        "between the two eyes) changes with yaw angle. A flat 2D photo or screen does not "
        "have this 3D property: IOD stays roughly constant even when the head appears to "
        "turn. High correlation = 3D depth = real person.",
    ),
    (
        "mean_fft_grid_peak",
        "📡",
        "Mean FFT grid peak — strength of the regular spatial grid pattern in the Fourier transform",
        "Screens emit light in a regular pixel grid. When a camera photographs or films a "
        "screen, the grid creates a distinctive peak in the image's Fourier transform (the "
        "same moiré pattern that causes rainbow patterns when you photograph a TV). This "
        "signal measures the average height of those grid peaks. High values indicate "
        "the camera is pointed at a screen rather than a real face.",
    ),
    (
        "interval_cv",
        "⏱️",
        "Interval CV — coefficient of variation of the inter-frame delivery intervals",
        "A genuine browser stream has small random jitter in the WebSocket delivery times "
        "because TCP scheduling and browser rendering are non-deterministic. A replay tool "
        "that injects frames programmatically delivers them at metronomically uniform "
        "intervals. CV = standard deviation ÷ mean of all inter-frame gaps. Very low CV "
        "(near 0) indicates a clock-driven injection rather than a live camera.",
    ),
    (
        "observed_fps",
        "🎞️",
        "Observed FPS — actual frames per second received at the server",
        "The browser is asked to send ~10 FPS. A genuine session delivers 8–12 FPS "
        "depending on CPU load and network. A heavily throttled replay or a very slow "
        "machine may deliver fewer frames. Combined with interval_cv, deviations from "
        "the expected 10 FPS help distinguish genuine sessions from programmatic injection.",
    ),
    (
        "frame_drop_rate",
        "📉",
        "Frame drop rate — fraction of expected frames that were not received",
        "If the session received significantly fewer frames than expected given its "
        "duration, that suggests the client was throttled, paused, or that frames were "
        "selectively dropped by a replay tool to reduce detection. High drop rates combined "
        "with low interval_cv are a suspicious combination.",
    ),
    (
        "mean_face_area_ratio",
        "🤳",
        "Mean face area ratio — average fraction of the frame filled by the face bounding box",
        "A genuine user sitting ~40–60 cm from their webcam fills roughly 15–35% of the "
        "frame with their face. A photo held very close to a phone camera may fill 70–90%. "
        "A photo held far away or a very small window may fill only 5%. Extreme values in "
        "either direction correlate with spoofing.",
    ),
    (
        "blur_risk_0_100",
        "🔵",
        "Blur risk — how blurry the face region is across all frames (0=sharp, 100=very blurry)",
        "A live face in motion has natural, frame-consistent blur during movement and sharp "
        "focus when still. A printed photo held at an angle, or a screen filmed from a bad "
        "angle, may produce unusual blur distributions. Very consistently high blur across "
        "all frames is suspicious.",
    ),
    (
        "brightness_risk_0_100",
        "☀️",
        "Brightness risk — how far the face lighting deviates from the expected range (0–100)",
        "A well-lit live face has a brightness in the 80–180 range (8-bit gray). An "
        "overexposed screen, an unlit room, or a very dark phone camera all push brightness "
        "outside that range. Combined with flicker_score, this helps detect screen-facing attacks.",
    ),
    (
        "wrong_action_count",
        "❌",
        "Wrong action count — number of challenges where the user performed the wrong action",
        "During challenges (blink, turn left, nod), the system detects whether the user "
        "completed the correct action. A replay video prepared for a different session "
        "will perform the wrong actions for this session's challenge sequence — because "
        "the attacker cannot know in advance which challenges will be issued. High counts "
        "indicate a pre-recorded video attack.",
    ),
    (
        "challenge_count",
        "🎯",
        "Challenge count — total number of challenges issued in this session",
        "Normally 3 challenges are issued per session. If the session ended early (timeout "
        "or error), fewer challenges were issued. The model learns that partial sessions "
        "have different risk profiles from complete sessions.",
    ),
    (
        "frames_without_face",
        "🚫",
        "Frames without face — total frames where no face was detected",
        "Occasional frames without a face are normal during head turns. A very high count "
        "suggests the attacker was using an intermittent or partial image rather than a "
        "live face directly facing the camera. Also useful for filtering out sessions "
        "where network issues caused excessive frame loss.",
    ),
    (
        "virtual_camera_suspected",
        "🖥️",
        "Virtual camera suspected — 1 if the browser reported a non-physical camera device",
        "The browser sends the camera device label when the user grants permission. "
        "OBS Virtual Camera, Snap Camera, and similar tools typically include words like "
        "'virtual', 'obs', 'snap', or 'screen' in their device label. This binary flag is "
        "set to 1 if the device label matched any of those patterns. It is the single "
        "most direct signal for virtual-camera injection attacks.",
    ),
]

# ─── API helpers ───────────────────────────────────────────────────────────────

@st.cache_resource
def _ensure_local_api() -> bool:
    """Start the FastAPI server if it is not already running locally.

    In Docker, BT_API_INTERNAL_URL is set so we skip auto-start.
    In local dev mode we spawn uvicorn once so the training WebSocket is available.
    Returns True once the port is accepting connections.
    """
    if os.getenv("BT_API_INTERNAL_URL"):
        return True  # Docker / explicit config — server managed externally

    def _port_open() -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", _API_PORT)) == 0

    if _port_open():
        return True

    # Spawn uvicorn in the background
    subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "basetruth.api:app",
            "--host", "127.0.0.1",
            "--port", str(_API_PORT),
            "--ws", "websockets-sansio",
            "--log-level", "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        if _port_open():
            return True
        time.sleep(0.5)
    return False


def _api_ws_url() -> str:
    """Return the WebSocket URL for the ML training endpoint."""
    internal = os.getenv("BT_API_INTERNAL_URL", f"http://localhost:{_API_PORT}")
    ws_base = internal.replace("https://", "wss://").replace("http://", "ws://")
    return f"{ws_base}/api/v1/ml/train/ws"


# ─── Background training thread ───────────────────────────────────────────────

def _training_thread_fn(
    models: List[str],
    msg_list: List[Dict],
    done_event: threading.Event,
    error_box: List[str],
) -> None:
    """Run in a daemon thread: connects to the FastAPI training WebSocket and
    reads all progress messages until the server sends 'all_done' or closes.

    msg_list and error_box are shared objects — Python's GIL makes list.append()
    effectively atomic for single-threaded appends, so no explicit lock is needed.
    """
    import asyncio  # noqa: PLC0415

    async def _run() -> None:
        import websockets  # noqa: PLC0415
        url = _api_ws_url()
        log.info("Training WebSocket client connecting", extra={"url": url})
        async with websockets.connect(url, ping_interval=None, open_timeout=15) as ws:
            # Tell the server which models to train
            await ws.send(json.dumps({"models": models}))
            async for msg_text in ws:
                try:
                    msg = json.loads(msg_text)
                except Exception:
                    continue
                msg_list.append(msg)
                # Stop reading once we've received the 'all_done' signal
                if msg.get("type") == "all_done":
                    break

    try:
        asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        error_box.append(str(exc))
        log.error("Training WebSocket error", extra={"error": str(exc)})
    finally:
        done_event.set()


# ─── Model status helpers ─────────────────────────────────────────────────────

def _get_model_status(model_type: str) -> Dict[str, Any]:
    """Return a dict describing the current state of a saved model file.

    Loads the pkl, reads sample counts from the CSV, and returns basic facts
    for the status cards at the top of the page.
    """
    if model_type == "face_scan_live":
        pkl_path = _LIVE_PKL
        csv_path = _LIVE_CSV
    elif model_type == "pdf":
        pkl_path = _PDF_PKL
        csv_path = _PDF_CSV
    else:
        pkl_path = _IMAGE_PKL
        csv_path = _IMAGE_CSV

    status: Dict[str, Any] = {
        "model_type": model_type,
        "model_exists": pkl_path.exists(),
        "csv_exists": csv_path.exists(),
        "pkl_path": str(pkl_path),
        "n_samples": 0,
        "n_features": 0,
        "last_modified": None,
        "class_counts": {},
    }

    if pkl_path.exists():
        try:
            import joblib  # noqa: PLC0415
            pipe = joblib.load(pkl_path)
            status["n_features"] = pipe.named_steps["model"].get_booster().num_features()
            status["last_modified"] = datetime.fromtimestamp(
                pkl_path.stat().st_mtime
            ).strftime("%d %b %Y, %H:%M")
        except Exception:
            pass

    if csv_path.exists():
        try:
            import pandas as pd  # noqa: PLC0415
            df = pd.read_csv(csv_path)
            status["n_samples"] = len(df)
            if "label" in df.columns:
                counts = df["label"].value_counts().to_dict()
                status["class_counts"] = {int(k): int(v) for k, v in counts.items()}
        except Exception:
            pass

    return status


# ─── Charts ───────────────────────────────────────────────────────────────────

def _draw_xgb_tree_mpl(
    booster: Any,
    feature_names: List[str],
    tree_idx: int = 0,
    max_depth: int = 5,
) -> Any:
    """Render one XGBoost tree as a matplotlib Figure — no graphviz required.

    XGBoost's built-in plot_tree() delegates to graphviz for layout. This
    function instead parses the JSON dump directly, assigns x/y positions
    via a post-order traversal (leaves get unique integer slots; internal nodes
    sit at the average of their children's positions), then draws everything
    with matplotlib FancyBboxPatch — indigo boxes for split nodes and
    green/red boxes for leaf nodes.  Edges are drawn first so boxes sit on top.

    Returns the Figure so the caller can st.pyplot() it and plt.close() it.
    """
    import json as _json  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.patches as mpatches  # noqa: PLC0415

    # ── 1. Parse the JSON dump ─────────────────────────────────────────────
    # get_dump() returns one JSON string per tree in the ensemble.
    raw_json = booster.get_dump(dump_format="json")
    if tree_idx >= len(raw_json):
        raise ValueError(
            f"Tree #{tree_idx} does not exist (ensemble has {len(raw_json)} trees)."
        )
    tree = _json.loads(raw_json[tree_idx])

    # Build a flat dict {node_id: node_data} by recursively walking the nested JSON.
    nodes: dict = {}

    def _parse(node: dict, depth: int = 0, parent: int = -1, branch: str = "") -> None:
        """Recursively flatten the nested tree JSON into the nodes dict."""
        nid = node["nodeid"]
        base: dict = {"depth": depth, "parent": parent, "branch": branch, "x": 0.0}
        if "leaf" in node:
            # Leaf node: log-odds score contribution toward the positive class.
            nodes[nid] = {**base, "type": "leaf", "value": float(node["leaf"])}
        else:
            # Split node: test feature < threshold, left=yes (<=), right=no (>).
            feat = node["split"]
            # XGBoost encodes feature names as 'f0', 'f1', etc. if names weren't
            # set on the DMatrix.  When they were set, the name is already readable.
            if feat.startswith("f") and feat[1:].isdigit():
                fidx = int(feat[1:])
                fname = feature_names[fidx] if fidx < len(feature_names) else feat
            else:
                fname = feat   # already a human-readable name from DMatrix
            nodes[nid] = {
                **base,
                "type": "split",
                "feature": fname,
                "threshold": float(node["split_condition"]),
                "yes": node["yes"],   # node_id for the left  (≤) branch
                "no":  node["no"],    # node_id for the right (>)  branch
            }
            if depth < max_depth:
                for child in node.get("children", []):
                    label = "yes" if child["nodeid"] == node["yes"] else "no"
                    _parse(child, depth + 1, nid, label)

    _parse(tree)

    # ── 2. Assign x positions via post-order traversal ─────────────────────
    # Each leaf gets a unique integer x slot; internal nodes centre over children.
    # leaf_x is a list so the nested closure can mutate it (Python 2-style trick).
    leaf_x: list = [0.0]

    def _assign_x(nid: int) -> float:
        """Assign an x coordinate to node nid and return it."""
        if nid not in nodes:
            return 0.0
        n = nodes[nid]
        if n["type"] == "leaf" or n["depth"] >= max_depth:
            # Leaf or pruned node: claim the next integer slot.
            n["x"] = float(leaf_x[0])
            leaf_x[0] += 1.0
            return n["x"]
        # Internal node: place at the mean of its two children's x values.
        child_ids = [n.get("yes"), n.get("no")]
        child_xs = [_assign_x(c) for c in child_ids if c in nodes]
        n["x"] = sum(child_xs) / len(child_xs) if child_xs else 0.0
        return n["x"]

    _assign_x(tree["nodeid"])

    # ── 3. Draw ────────────────────────────────────────────────────────────
    total_leaves = int(leaf_x[0]) or 1
    max_d = max((n["depth"] for n in nodes.values()), default=1) + 1

    # Scale figure so small trees don't look cramped and huge trees don't overflow.
    xs = max(1.8, 18.0 / total_leaves)   # horizontal spacing in inches per leaf slot
    ys = 1.6                              # vertical spacing in inches per depth level
    node_w = min(1.5, xs * 0.9)          # node box width (capped so text fits)
    node_h = 0.52                         # node box height (fixed)

    fig, ax = plt.subplots(
        figsize=(total_leaves * xs, max_d * ys), facecolor="#0f172a"
    )
    ax.set_facecolor("#0f172a")
    ax.set_xlim(-node_w, total_leaves * xs)
    ax.set_ylim(-node_h * 2, max_d * ys)
    ax.invert_yaxis()   # depth=0 (root) at top, leaves at bottom
    ax.axis("off")

    def _node_xy(n: dict) -> tuple:
        """Convert logical (x-slot, depth) to figure data coordinates."""
        return n["x"] * xs, n["depth"] * ys

    # Draw edges first so they appear behind node boxes.
    for nid, n in nodes.items():
        if n["type"] == "split" and n["depth"] < max_depth:
            for child_id, lbl in [(n.get("yes"), "≤"), (n.get("no"), ">")]:
                if child_id in nodes:
                    c = nodes[child_id]
                    px, py = _node_xy(n)
                    cx, cy = _node_xy(c)
                    ax.annotate(
                        "",
                        xy=(cx, cy - node_h / 2),
                        xytext=(px, py + node_h / 2),
                        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.2),
                        annotation_clip=False,
                    )
                    # Branch label (≤ or >) at the midpoint of each edge.
                    mx, my = (px + cx) / 2, (py + cy) / 2
                    ax.text(
                        mx, my, lbl, ha="center", va="center",
                        color="#94a3b8", fontsize=7, fontweight="bold",
                        bbox=dict(facecolor="#0f172a", edgecolor="none", pad=1),
                    )

    # Draw node boxes on top of edges.
    for nid, n in nodes.items():
        x, y = _node_xy(n)
        if n["type"] == "split" and n["depth"] < max_depth:
            # Shorten long feature names so they fit inside the node box.
            fname = (
                n["feature"]
                .replace("render_", "rnd_")
                .replace("_suspicious", "_susp")
                .replace("_hotspot", "_hot")
                .replace("_ratio", "_r")
                .replace("_score", "_s")
                .replace("_count", "_n")
            )
            label = f"{fname}\n< {n['threshold']:.3g}"
            fc, ec = "#1e3a5f", "#6366f1"   # indigo split node
        else:
            # Leaf or depth-pruned node — colour by sign of log-odds contribution.
            val = n.get("value", 0.0)
            label = f"leaf\n{val:+.4f}"
            fc = "#14532d" if val >= 0 else "#7f1d1d"   # green=positive, red=negative
            ec = "#4ade80" if val >= 0 else "#f87171"

        box = mpatches.FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2),
            node_w, node_h,
            boxstyle="round,pad=0.05",
            facecolor=fc, edgecolor=ec, linewidth=1.5,
            transform=ax.transData, clip_on=False,
        )
        ax.add_patch(box)
        ax.text(
            x, y, label, ha="center", va="center",
            color="white", fontsize=7, fontweight="bold",
            multialignment="center", linespacing=1.3,
        )

    ax.set_title(
        f"XGBoost Tree #{tree_idx}  "
        f"(top {max_depth} levels · left branch = YES/≤ · right branch = NO/>)",
        color="white", fontsize=10, pad=8,
    )
    plt.tight_layout(pad=0.4)
    return fig


def _build_image_charts(results: Dict[str, Any]) -> None:
    """Render the four post-training analytics charts for the Image model.

    Each chart is shown inside its own Streamlit expander so the page stays
    compact.  All charts use a matching dark navy theme.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np
        import pandas as pd
        import joblib
        import warnings
        warnings.filterwarnings("ignore")
    except ImportError as e:
        st.warning(f"matplotlib / pandas not available for charts: {e}")
        return

    if not _IMAGE_CSV.exists() or not _IMAGE_PKL.exists():
        return

    df = pd.read_csv(_IMAGE_CSV)
    pipe = joblib.load(_IMAGE_PKL)

    booster = pipe.named_steps["model"].get_booster()
    n_booster = booster.num_features()

    from basetruth.analysis.ml_scorer import FEATURE_NAMES, _remap_raw_csv  # noqa: PLC0415
    active_names = FEATURE_NAMES[:n_booster]

    # ── Panel A: Class Distribution ──────────────────────────────────────────
    with st.expander("📊  Chart A — How is the training data split?", expanded=True):
        st.markdown(
            "Each bar shows how many images are in each verdict category. "
            "**Balanced classes** (similar bar heights) make a more reliable model.",
        )
        if "label" in df.columns:
            # Binarize: 0=ORIGINAL, 1=FAKE/EDITED (labels 1,2,3 collapsed)
            binary_labels = (df["label"] > 0).astype(int)
            counts = [int((binary_labels == i).sum()) for i in range(2)]
            fig_a, ax_a = plt.subplots(figsize=(7, 2.5), facecolor="#0f172a")
            ax_a.set_facecolor("#1e293b")
            bars = ax_a.barh(_CLASS_LABELS, counts, color=_CLASS_COLORS, edgecolor="#334155", height=0.5)
            for bar, cnt in zip(bars, counts):
                ax_a.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                          str(cnt), va="center", ha="left", color="white", fontweight="bold", fontsize=10)
            ax_a.set_xlabel("Number of training images", color="white")
            ax_a.tick_params(colors="white")
            ax_a.set_facecolor("#1e293b")
            ax_a.invert_yaxis()
            for spine in ax_a.spines.values():
                spine.set_edgecolor("#334155")
            ax_a.grid(axis="x", color="#334155", linewidth=0.6, alpha=0.5)
            ax_a.set_xlim(0, max(counts) * 1.2)
            ax_a.set_title("Training Class Distribution (Binary)", color="white", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_a, use_container_width=True)
            plt.close(fig_a)

    # ── Panel B: Feature Importance ───────────────────────────────────────────
    with st.expander("🏆  Chart B — Which signals matter most? (Global importance)"):
        st.markdown(
            "**Gain importance** measures how much each signal reduced errors when the model "
            "chose it as a decision point. Higher bar = more influential signal. "
            "Signals with zero bars were never helpful enough to use.",
        )
        raw_scores = booster.get_score(importance_type="gain")
        imp_vals = [raw_scores.get(f"f{i}", 0.0) for i in range(n_booster)]
        display_names = [
            n.replace("_score", "").replace("_ratio", "").replace("_count", "").replace("_", " ").title()
            for n in active_names
        ]
        # Sort highest first
        order = sorted(range(len(imp_vals)), key=lambda i: imp_vals[i])
        sorted_vals  = [imp_vals[i] for i in order]
        sorted_names = [display_names[i] for i in order]

        cmap = LinearSegmentedColormap.from_list("rg", ["#22c55e", "#ef4444"])
        bar_colors = [cmap(v / max(sorted_vals) if max(sorted_vals) > 0 else 0) for v in sorted_vals]

        fig_b, ax_b = plt.subplots(figsize=(7, max(3, len(active_names) * 0.45)), facecolor="#0f172a")
        ax_b.set_facecolor("#1e293b")
        ax_b.barh(range(len(sorted_vals)), sorted_vals, color=bar_colors, edgecolor="#334155", height=0.7)
        ax_b.set_yticks(range(len(sorted_names)))
        ax_b.set_yticklabels(sorted_names, fontsize=9, color="white")
        ax_b.set_xlabel("Mean Gain (higher = more influential)", color="white")
        ax_b.tick_params(colors="white")
        ax_b.invert_yaxis()
        for spine in ax_b.spines.values():
            spine.set_edgecolor("#334155")
        ax_b.grid(axis="x", color="#334155", linewidth=0.6, alpha=0.5)
        ax_b.set_title("Signal Importance (Gain)", color="white", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)

    # ── Panel C: Per-Class SHAP Heatmap ───────────────────────────────────────
    with st.expander("🔬  Chart C — Which signals drive each verdict? (SHAP heatmap)"):
        st.markdown(
            "Each cell shows how strongly a signal pushes the model toward that verdict on "
            "average across all training samples. **Bright = strong influence**, "
            "dark = weak influence. Compare rows to see which signals are unique to each class.",
        )
        try:
            from basetruth.analysis.ml_scorer import _remap_raw_csv  # noqa: PLC0415
            import xgboost as xgb  # noqa: PLC0415

            df_feat = _remap_raw_csv(df)
            X = df_feat[FEATURE_NAMES].copy().astype(float)
            X.replace(-1.0, float("nan"), inplace=True)
            # No extra columns — model was trained with exactly FEATURE_NAMES

            imputer = pipe.named_steps["imputer"]
            X_imp = imputer.transform(X)
            X_trim = X_imp[:, :n_booster]
            dmat = xgb.DMatrix(X_trim, feature_names=active_names)

            # Binary model: pred_contribs returns 2-D (n_samples, n_features+1)
            raw_shap = booster.predict(dmat, pred_contribs=True)
            # Drop bias column; result is (n_samples, n_features)
            shap_2d = raw_shap[:, :-1]

            # Stack into shape (2, n_features) — row 0 = ORIGINAL, row 1 = FAKE/EDITED
            # For binary XGBoost, positive SHAP → pushes toward FAKE/EDITED
            mean_abs_fake = np.mean(np.abs(shap_2d), axis=0, keepdims=True)   # (1, n_feat)
            mean_abs_orig = mean_abs_fake  # symmetric for binary logistic
            mean_abs = np.vstack([mean_abs_orig, mean_abs_fake])  # (2, n_feat)

            # Column-normalise so colours reflect relative within-feature importance
            col_max = mean_abs.max(axis=0, keepdims=True)
            col_max[col_max == 0] = 1.0
            normalised = mean_abs / col_max

            short_names = [
                n.replace("_score", "").replace("_ratio", "").replace("_count", "")
                 .replace("_", "\n").title()
                for n in active_names
            ]

            fig_c, ax_c = plt.subplots(
                figsize=(max(8, n_booster * 0.9), 2.5),
                facecolor="#0f172a",
            )
            ax_c.set_facecolor("#1e293b")

            heat_cmap = LinearSegmentedColormap.from_list("heat", ["#0f172a", "#f59e0b", "#ef4444"])
            im = ax_c.imshow(normalised, aspect="auto", cmap=heat_cmap, vmin=0, vmax=1)

            # Annotate each cell with the raw mean |SHAP| value
            for row in range(normalised.shape[0]):
                for col in range(normalised.shape[1]):
                    val = mean_abs[row, col]
                    text_col = "black" if normalised[row, col] > 0.65 else "white"
                    ax_c.text(col, row, f"{val:.3f}", ha="center", va="center",
                              fontsize=7, color=text_col, fontweight="bold")

            ax_c.set_xticks(range(len(active_names)))
            ax_c.set_xticklabels(short_names, fontsize=8, color="white")
            ax_c.set_yticks(range(2))
            ax_c.set_yticklabels(_CLASS_LABELS, fontsize=9, color="white")
            ax_c.set_title("Per-Verdict SHAP Influence (column-normalised)", color="white", fontsize=11)

            cbar = plt.colorbar(im, ax=ax_c, fraction=0.015, pad=0.01)
            cbar.set_label("Relative influence", color="white", fontsize=8)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

            plt.tight_layout()
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)

        except Exception as e:
            st.warning(f"SHAP chart unavailable: {e}")

    # ── Panel D: Confusion Matrix ─────────────────────────────────────────────
    with st.expander("✅  Chart D — Confusion matrix (how often is the model right?)"):
        st.markdown(
            "Rows = actual verdict, Columns = predicted verdict. "
            "The diagonal (top-left → bottom-right) shows **correct predictions**. "
            "Off-diagonal cells show where the model made mistakes.",
        )
        try:
            from sklearn.metrics import confusion_matrix  # noqa: PLC0415
            from basetruth.analysis.ml_scorer import _remap_raw_csv  # noqa: PLC0415

            df_feat = _remap_raw_csv(df)
            X = df_feat[FEATURE_NAMES].copy().astype(float)
            X.replace(-1.0, float("nan"), inplace=True)
            # Binarize labels to match the trained model
            y_true = (df["label"].values.astype(int) > 0).astype(int)
            y_pred = pipe.predict(X)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            # Normalise rows → each cell = recall for that class
            with np.errstate(divide="ignore", invalid="ignore"):
                cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                cm_norm = np.nan_to_num(cm_norm)

            fig_d, ax_d = plt.subplots(figsize=(4.5, 3.5), facecolor="#0f172a")
            ax_d.set_facecolor("#1e293b")
            cmap_cm = LinearSegmentedColormap.from_list("cm_heat", ["#0f172a", "#22c55e"])
            im_d = ax_d.imshow(cm_norm, aspect="auto", cmap=cmap_cm, vmin=0, vmax=1)

            for i in range(2):
                for j in range(2):
                    text_col = "black" if cm_norm[i, j] > 0.6 else "white"
                    ax_d.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                              ha="center", va="center", fontsize=10, color=text_col)

            ax_d.set_xticks(range(2))
            ax_d.set_xticklabels([f"Pred\n{l}" for l in _CLASS_LABELS], fontsize=9, color="white")
            ax_d.set_yticks(range(2))
            ax_d.set_yticklabels([f"True\n{l}" for l in _CLASS_LABELS], fontsize=9, color="white")
            ax_d.set_title("Confusion Matrix (row-normalised)", color="white", fontsize=11)

            cbar_d = plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
            cbar_d.set_label("Recall (% of actual class correct)", color="white", fontsize=8)
            cbar_d.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar_d.ax.yaxis.get_ticklabels(), color="white")
            plt.tight_layout()
            st.pyplot(fig_d, use_container_width=True)
            plt.close(fig_d)
        except Exception as e:
            st.warning(f"Confusion matrix unavailable: {e}")

    # ── Panel E: PCA Scatter ─────────────────────────────────────────────
    with st.expander("🌍  Chart E — Where does each document land? (PCA scatter)"):
        st.markdown(
            "Each dot is one training document. The two axes are not real signals — they are "
            "the two **principal components** (the directions of maximum variance across all "
            f"{len(active_names)} signals combined). Clusters far apart are well-separated "
            "in the model's feature space; overlapping clusters are harder to distinguish."
        )
        try:
            from sklearn.decomposition import PCA  # noqa: PLC0415
            from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

            df_pca = pd.read_csv(_IMAGE_CSV)
            # Keep only rows where every active feature is present.
            df_pca_feat = df_pca[active_names].copy().astype(float)
            df_pca_feat.replace(-1.0, float("nan"), inplace=True)
            df_pca_feat.fillna(df_pca_feat.median(), inplace=True)
            # No extra columns — active_names is exactly what the model was trained with

            # Standardise so no single large-valued feature dominates PCA.
            X_scaled = StandardScaler().fit_transform(df_pca_feat[active_names])
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            var_exp = pca.explained_variance_ratio_

            # Binarize labels for PCA colouring
            y_lbl = (df_pca["label"].values.astype(int) > 0).astype(int)

            fig_e, ax_e = plt.subplots(figsize=(7, 5), facecolor="#0f172a")
            ax_e.set_facecolor("#1e293b")
            for cls_idx, (cls_name, cls_col) in enumerate(zip(_CLASS_LABELS, _CLASS_COLORS)):
                mask = y_lbl == cls_idx
                if mask.any():
                    ax_e.scatter(
                        coords[mask, 0], coords[mask, 1],
                        c=cls_col, label=cls_name, alpha=0.7, s=18, edgecolors="none",
                    )
            ax_e.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)", color="white", fontsize=9)
            ax_e.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)", color="white", fontsize=9)
            ax_e.set_title("Document Feature Space (PCA 2-D)", color="white", fontsize=11)
            ax_e.tick_params(colors="white")
            for spine in ax_e.spines.values():
                spine.set_edgecolor("#334155")
            leg_e = ax_e.legend(fontsize=8, framealpha=0.2, labelcolor="white", facecolor="#0f172a")
            leg_e.get_frame().set_edgecolor("#334155")
            plt.tight_layout()
            st.pyplot(fig_e, use_container_width=True)
            plt.close(fig_e)
            st.caption(
                f"Total variance explained by 2 components: "
                f"**{sum(var_exp):.1%}**. "
                "The remaining variance is spread across the other dimensions."
            )
        except Exception as e:
            st.warning(f"PCA scatter unavailable: {e}")

    # ── Panel F: XGBoost Decision Tree ─────────────────────────────────────
    with st.expander("🌲  Chart F — How does the model actually make a decision? (Tree #0)"):
        st.markdown(
            "XGBoost trains **hundreds of trees** that vote together. This shows **Tree\u2060 #0** — "
            "the very first tree the model built. It is the single most important split sequence. "
            "Each node shows the signal it tests and the threshold value. Left branch = ≤ threshold, "
            "right branch = > threshold. Leaf values are raw score contributions (log-odds)."
        )
        try:
            # Use the pure-matplotlib renderer — no graphviz dependency needed.
            from basetruth.analysis.ml_scorer import FEATURE_NAMES as _img_fn  # noqa: PLC0415
            fig_f = _draw_xgb_tree_mpl(booster, list(_img_fn), tree_idx=0, max_depth=5)
            st.pyplot(fig_f, use_container_width=True)
            plt.close(fig_f)
            st.caption(
                "⚠️ The full model uses many trees. This single tree is illustrative only — "
                "its leaf values only become meaningful scores when combined with all other trees."
            )
        except Exception as e:
            st.warning(f"Decision tree visualisation unavailable: {e}")


def _build_pdf_charts(results: Dict[str, Any]) -> None:
    """Render feature importance chart for the PDF binary model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import joblib
        import warnings
        warnings.filterwarnings("ignore")
    except ImportError:
        return

    if not _PDF_PKL.exists():
        return

    pipe = joblib.load(_PDF_PKL)
    booster = pipe.named_steps["model"].get_booster()
    n_booster = booster.num_features()

    from basetruth.analysis.ml_scorer_pdf import PDF_FEATURE_NAMES  # noqa: PLC0415
    active_names = PDF_FEATURE_NAMES[:n_booster]

    with st.expander("🏆  Chart — Which PDF signals matter most?", expanded=True):
        st.markdown(
            "Gain importance for the PDF fraud model. Each bar is a forensic signal "
            "extracted from inside the PDF file (not the rendered image).",
        )
        raw_scores = booster.get_score(importance_type="gain")
        imp_vals = [raw_scores.get(f"f{i}", 0.0) for i in range(n_booster)]
        display_names = [
            n.replace("_score", "").replace("_ratio", "").replace("_count", "")
             .replace("_", " ").title()
            for n in active_names
        ]
        order = sorted(range(len(imp_vals)), key=lambda i: imp_vals[i])
        sorted_vals  = [imp_vals[i] for i in order]
        sorted_names = [display_names[i] for i in order]

        cmap = LinearSegmentedColormap.from_list("rg", ["#22c55e", "#ef4444"])
        bar_colors = [cmap(v / max(sorted_vals) if max(sorted_vals) > 0 else 0) for v in sorted_vals]

        fig, ax = plt.subplots(figsize=(7, max(3, n_booster * 0.4)), facecolor="#0f172a")
        ax.set_facecolor("#1e293b")
        ax.barh(range(len(sorted_vals)), sorted_vals, color=bar_colors, edgecolor="#334155", height=0.7)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=8, color="white")
        ax.set_xlabel("Mean Gain", color="white")
        ax.tick_params(colors="white")
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.grid(axis="x", color="#334155", linewidth=0.6, alpha=0.5)
        ax.set_title("PDF Signal Importance (Gain)", color="white", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── PCA Scatter ──────────────────────────────────────────────────────────
    with st.expander("🌍  Chart — Where does each PDF land? (PCA scatter)"):
        st.markdown(
            "Each dot is one training PDF. The two axes are the **principal components** of all "
            f"{n_booster} PDF signals combined. Dots with the same colour were all labelled the "
            "same class. Tight, well-separated clusters mean the model has a clean boundary "
            "to learn; overlapping blobs indicate harder-to-classify documents."
        )
        try:
            import pandas as pd  # noqa: PLC0415
            from sklearn.decomposition import PCA  # noqa: PLC0415
            from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

            _pdf_colours = ["#22c55e", "#ef4444"]  # green=original, red=tampered
            _pdf_labels  = ["ORIGINAL (0)", "TAMPERED (1)"]

            df_pca = pd.read_csv(_PDF_CSV)
            feat_cols = [c for c in active_names if c in df_pca.columns]
            df_pca_feat = df_pca[feat_cols].copy().astype(float)
            df_pca_feat.fillna(df_pca_feat.median(), inplace=True)

            X_scaled = StandardScaler().fit_transform(df_pca_feat)
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            var_exp = pca.explained_variance_ratio_
            y_lbl = df_pca["label"].values.astype(int)

            fig_p, ax_p = plt.subplots(figsize=(7, 5), facecolor="#0f172a")
            ax_p.set_facecolor("#1e293b")
            for cls_idx, (cls_name, cls_col) in enumerate(zip(_pdf_labels, _pdf_colours)):
                mask = y_lbl == cls_idx
                if mask.any():
                    ax_p.scatter(
                        coords[mask, 0], coords[mask, 1],
                        c=cls_col, label=cls_name, alpha=0.7, s=18, edgecolors="none",
                    )
            ax_p.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)", color="white", fontsize=9)
            ax_p.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)", color="white", fontsize=9)
            ax_p.set_title("PDF Document Feature Space (PCA 2-D)", color="white", fontsize=11)
            ax_p.tick_params(colors="white")
            for spine in ax_p.spines.values():
                spine.set_edgecolor("#334155")
            leg_p = ax_p.legend(fontsize=8, framealpha=0.2, labelcolor="white", facecolor="#0f172a")
            leg_p.get_frame().set_edgecolor("#334155")
            plt.tight_layout()
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)
            st.caption(
                f"Total variance explained: **{sum(var_exp):.1%}**. "
                "Remaining variance is distributed across the other dimensions."
            )
        except Exception as e:
            st.warning(f"PCA scatter unavailable: {e}")

    # ── XGBoost Decision Tree ─────────────────────────────────────────────────
    with st.expander("🌲  Chart — How does the PDF model decide? (Tree #0)"):
        st.markdown(
            "XGBoost trains **many trees** that vote together. This shows **Tree\u2060 #0** — "
            "the single most important split sequence. Each node shows the PDF signal it "
            "tests and the threshold value. Left branch = ≤ threshold, right = > threshold. "
            "Leaf values are raw score contributions (log-odds toward TAMPERED)."
        )
        try:
            # Use the pure-matplotlib renderer — no graphviz dependency needed.
            from basetruth.analysis.ml_scorer_pdf import PDF_FEATURE_NAMES as _pdf_fn  # noqa: PLC0415
            fig_t = _draw_xgb_tree_mpl(booster, list(_pdf_fn), tree_idx=0, max_depth=5)
            st.pyplot(fig_t, use_container_width=True)
            plt.close(fig_t)
            st.caption(
                "⚠️ This is one tree from a large ensemble. Its leaf values are only meaningful "
                "when summed with all other trees in the model."
            )
        except Exception as e:
            st.warning(f"Decision tree visualisation unavailable: {e}")


# ─── CSS ──────────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* Status card grid */
.mlt-status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
.mlt-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
}
.mlt-card-title  { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
.mlt-card-value  { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; margin-top: 0.1rem; }
.mlt-card-sub    { font-size: 0.78rem; color: #64748b; margin-top: 0.15rem; }
.mlt-card.good   { border-color: rgba(34,197,94,0.4);  }
.mlt-card.warn   { border-color: rgba(234,179,8,0.4);  }
.mlt-card.bad    { border-color: rgba(239,68,68,0.35); }
/* Progress log */
.mlt-log-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: monospace;
    font-size: 0.8rem;
    max-height: 320px;
    overflow-y: auto;
}
.mlt-log-line { display: flex; gap: 0.8rem; margin-bottom: 0.25rem; align-items: flex-start; }
.mlt-log-ts   { color: #475569; white-space: nowrap; flex-shrink: 0; }
.mlt-log-tag  { font-weight: 700; white-space: nowrap; flex-shrink: 0; }
.mlt-log-step { color: #cbd5e1; }
.mlt-log-done { color: #22c55e; }
.mlt-log-err  { color: #ef4444; }
/* Signal guide — card-based layout */
.sig-card {
    display: flex;
    gap: 1.1rem;
    align-items: flex-start;
    background: #1e293b;
    border: 1px solid #334155;
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    transition: border-left-color 0.2s;
}
.sig-card:hover { border-left-color: #818cf8; }
.sig-card.sig-inactive { opacity: 0.45; filter: grayscale(0.4); }
/* Large centred emoji bubble */
.sig-emoji-wrap {
    flex-shrink: 0;
    width: 2.8rem;
    height: 2.8rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(99,102,241,0.12);
    border-radius: 50%;
    font-size: 1.45rem;
    line-height: 1;
}
/* Right column */
.sig-content { flex: 1; min-width: 0; }
/* Signal name in a monospace chip */
.sig-name-chip {
    display: inline-block;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 0.78rem;
    font-weight: 700;
    color: #a5b4fc;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 5px;
    padding: 1px 7px;
    letter-spacing: 0.02em;
    margin-bottom: 0.35rem;
}
/* One-line summary */
.sig-summary {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.4rem;
    line-height: 1.35;
}
/* Body explanation */
.sig-detail {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.65;
    margin: 0;
}
/* 'NOT IN MODEL' badge */
.sig-stale-badge {
    display: inline-block;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 2px 7px;
    border-radius: 4px;
    background: rgba(239,68,68,0.15);
    color: #fca5a5;
    border: 1px solid rgba(239,68,68,0.25);
    margin-left: 8px;
    vertical-align: middle;
}
</style>
"""


# ─── Log rendering helper ─────────────────────────────────────────────────────

def _render_log(msg_list: List[Dict]) -> None:
    """Render the live WebSocket training log as a styled monospace feed."""
    lines = []
    for msg in msg_list:
        t = msg.get("type", "log")
        model = msg.get("model", "")
        step  = msg.get("step", msg.get("message", ""))
        ts    = datetime.now().strftime("%H:%M:%S")
        tag_color = "#3b82f6" if model == "pdf" else "#a855f7"

        if t == "log":
            pct = msg.get("pct", 0)
            lines.append(
                f'<div class="mlt-log-line">'
                f'<span class="mlt-log-ts">{ts}</span>'
                f'<span class="mlt-log-tag" style="color:{tag_color}">[{model.upper()}]</span>'
                f'<span class="mlt-log-step">{step}  <span style="color:#475569">({pct}%)</span></span>'
                f'</div>'
            )
        elif t == "done":
            m = msg.get("metrics", {})
            summary = (
                f"✅ Done — accuracy {m.get('accuracy', 0):.1%}  |  "
                f"F1 {m.get('f1', 0):.1%}  |  AUC {m.get('roc_auc', 0):.3f}"
            )
            lines.append(
                f'<div class="mlt-log-line">'
                f'<span class="mlt-log-ts">{ts}</span>'
                f'<span class="mlt-log-tag" style="color:{tag_color}">[{model.upper()}]</span>'
                f'<span class="mlt-log-done">{summary}</span>'
                f'</div>'
            )
        elif t == "error":
            lines.append(
                f'<div class="mlt-log-line">'
                f'<span class="mlt-log-ts">{ts}</span>'
                f'<span class="mlt-log-tag" style="color:#ef4444">[{model.upper()} ERROR]</span>'
                f'<span class="mlt-log-err">{step}</span>'
                f'</div>'
            )
        elif t == "all_done":
            lines.append(
                f'<div class="mlt-log-line">'
                f'<span class="mlt-log-ts">{ts}</span>'
                f'<span class="mlt-log-done">🎉 All training jobs complete.</span>'
                f'</div>'
            )

    log_html = '<div class="mlt-log-box">' + "".join(lines) + "</div>"
    st.markdown(log_html, unsafe_allow_html=True)


# ─── Metric cards ─────────────────────────────────────────────────────────────

def _metric_card(title: str, value: str, sub: str = "", quality: str = "good") -> str:
    """Return an HTML metric card string."""
    return (
        f'<div class="mlt-card {quality}">'
        f'<div class="mlt-card-title">{title}</div>'
        f'<div class="mlt-card-value">{value}</div>'
        f'<div class="mlt-card-sub">{sub}</div>'
        f'</div>'
    )


def _render_metrics_cards(metrics: Dict[str, Any], model_type: str) -> None:
    """Render accuracy / F1 / AUC metric cards for a trained model."""
    acc  = metrics.get("accuracy", 0)
    f1   = metrics.get("f1", 0)
    auc  = metrics.get("roc_auc", 0)
    rows = metrics.get("rows_trained", 0)
    label = "Image" if model_type == "image" else "PDF"

    def _q(v: float) -> str:
        return "good" if v >= 0.90 else ("warn" if v >= 0.80 else "bad")

    cards_html = (
        '<div class="mlt-status-grid">'
        + _metric_card("Accuracy", f"{acc:.1%}", f"{label} model — {rows} samples", _q(acc))
        + _metric_card("F1 Score", f"{f1:.1%}", "Weighted across all classes", _q(f1))
        + _metric_card("ROC AUC", f"{auc:.3f}", "1.0 = perfect; 0.5 = random guess", _q(auc))
        + _metric_card(
            "Training samples", str(rows),
            "Samples used for cross-validation",
            "good" if rows >= 100 else "warn",
        )
        + "</div>"
    )
    st.markdown(cards_html, unsafe_allow_html=True)

    with st.expander("What do these numbers mean?"):
        st.markdown(
            "**Accuracy** — out of all documents in the test fold, what percentage did the "
            "model assign to the correct verdict class? 95% means 19 out of 20 were correct.\n\n"
            "**F1 Score** — a balanced measure that penalises both false positives (wrongly "
            "flagging genuine documents) AND false negatives (missing real tampered ones). "
            "It is especially useful when the classes have different sizes.\n\n"
            "**ROC AUC** — Area Under the Curve: measures how well the model can rank "
            "tampered documents above genuine ones. 1.0 = perfect ranking. 0.5 = the model "
            "is no better than a coin flip. Anything above 0.85 is considered strong.\n\n"
            "These scores come from **5-fold cross-validation**, meaning the data was split "
            "5 times and each time the model was tested on data it had never seen during "
            "training — a rigorous way to check for overfitting."
        )


# ─── Signal guide section ─────────────────────────────────────────────────────

def _render_signal_guide(
    guide: list,
    active_names,
    title: str,
    intro: str,
) -> None:
    """Render a plain-English signal guide for any forensic feature list.

    Each signal is shown as a standalone card — emoji bubble on the left,
    monospace name chip + one-line summary + full explanation on the right.
    Cards for signals not present in the currently saved model are greyed out.

    guide        — list of (name, emoji, summary, detail) tuples
    active_names — set[str] of feature names in the saved model, or None to
                   show all cards active (no model trained yet)
    title        — section heading text (without the markdown prefix)
    intro        — one-sentence explanation shown below the heading
    """
    n = len(guide)
    st.markdown(f"#### 🔍 {title}")
    st.markdown(intro)

    # Group signals into 2 columns so the list doesn't become an endless single column.
    col_left, col_right = st.columns(2, gap="medium")
    half = (n + 1) // 2   # ceiling division → left col gets the extra card when n is odd

    for col, signal_slice in [
        (col_left,  guide[:half]),
        (col_right, guide[half:]),
    ]:
        with col:
            for _i, (name, emoji, summary, detail) in enumerate(signal_slice):
                # Name-based greyout: None means all active (no model saved yet).
                # This is correct regardless of how many entries the guide has vs.
                # how many features the model uses — positional index would be wrong.
                in_model = active_names is None or name in active_names
                inactive_class = "" if in_model else " sig-inactive"
                stale_badge = (
                    ""
                    if in_model
                    else '<span class="sig-stale-badge">NOT IN SAVED MODEL</span>'
                )

                # Render a self-contained card — no expander, everything visible.
                # The card uses a coloured left border so it reads like a structured list.
                st.markdown(
                    f"<div class='sig-card{inactive_class}'>"
                    f"  <div class='sig-emoji-wrap'>{emoji}</div>"
                    f"  <div class='sig-content'>"
                    f"    <div class='sig-name-chip'>{name}{stale_badge}</div>"
                    f"    <div class='sig-summary'>{summary.split(' — ', 1)[-1].capitalize()}</div>"
                    f"    <p class='sig-detail'>{detail}</p>"
                    f"  </div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ─── Status cards for the header section ──────────────────────────────────────

def _render_status_section() -> None:
    """Show current model file status for both image and PDF models."""
    img_st  = _get_model_status("image")
    pdf_st  = _get_model_status("pdf")
    live_st = _get_model_status("face_scan_live")

    def _card(s: Dict, label: str) -> str:
        if s["model_exists"]:
            n  = s["n_samples"]
            nf = s["n_features"]
            ts = s["last_modified"] or "unknown"
            return _metric_card(
                f"{label} Model",
                "✅ Trained",
                f"{n} samples · {nf} signals · last updated {ts}",
                "good",
            )
        elif s["csv_exists"]:
            n = s["n_samples"]
            return _metric_card(
                f"{label} Model",
                "⚠️ Not trained",
                f"Training CSV found ({n} samples) — click Start Training to build",
                "warn",
            )
        else:
            return _metric_card(
                f"{label} Model",
                "❌ No data",
                "Training CSV not found — run collect_live_scan_samples.py first",
                "bad",
            )

    st.markdown(
        '<div class="mlt-status-grid">'
        + _card(img_st,  "Image")
        + _card(pdf_st,  "PDF")
        + _card(live_st, "Face Scan Live")
        + "</div>",
        unsafe_allow_html=True,
    )

    # Signal coverage warning — only show when an old model (< 19 features) is loaded
    if img_st["model_exists"] and img_st["n_features"] < 19:
        missing = 19 - img_st["n_features"]
        st.warning(
            f"⚠️  The current **Image model** was trained on **{img_st['n_features']} of 19 signals** "
            f"({missing} signal{'s' if missing > 1 else ''} added after the last training run). "
            "Retraining will include all 19 signals.",
            icon="⚠️",
        )


# ─── Data Extraction helpers ──────────────────────────────────────────────────
# These helpers drive the "Data Extraction" tab: browsing sample folders,
# kicking off the WebSocket-based per-file forensics collection, live progress
# display, and 4 post-extraction analytics charts.

_SAMPLE_ROOT = _REPO_ROOT / "fraud_model" / "sample"
# Supported image extensions (same set as the collection script).
_IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# All 6 input sample folders listed in pipeline order.
# Tuple: (data_type, folder_name, label_int, label_str, description)
_SAMPLE_FOLDER_DEFS: List[tuple] = [
    ("images", "original_images",         0, "ORIGINAL",         "Phone-fresh, unmodified genuine documents."),
    ("images", "original_derived_images", 1, "ORIGINAL-DERIVED", "Save-as copies of genuine docs (still clean, but carries JPEG re-compression artefacts)."),
    ("images", "tampered_images",         2, "TAMPERED",         "Directly forged / manipulated images — strongest forensic signals."),
    ("images", "tampered_derived_images", 3, "TAMPERED-DERIVED", "Save-as copies of tampered images — laundered fraud with softer ELA/clone signals."),
    ("pdfs",   "original_pdfs",           0, "ORIGINAL",         "Genuine PDFs — payroll-generated or otherwise pristine."),
    ("pdfs",   "tampered_pdfs",           1, "TAMPERED",         "Manipulated PDFs — text replacement, added fields, etc."),
]


def _get_sample_folder_stats() -> List[Dict]:
    """Return file counts and existence flags for all 6 sample input folders.

    The result is used both for the folder browser table and for Chart A.
    """
    rows = []
    for dtype, fname, label_int, label_str, desc in _SAMPLE_FOLDER_DEFS:
        fp = _SAMPLE_ROOT / fname
        if fp.is_dir():
            if dtype == "images":
                count = sum(1 for p in fp.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
            else:
                count = sum(1 for p in fp.iterdir() if p.suffix.lower() == ".pdf")
            exists = True
        else:
            count  = 0
            exists = False
        rows.append({
            "data_type":   dtype,
            "folder":      fname,
            "label":       label_str,
            "label_int":   label_int,
            "files":       count,
            "exists":      exists,
            "description": desc,
        })
    return rows


def _extraction_ws_url() -> str:
    """Return the WebSocket URL for the ML feature-extraction endpoint."""
    internal = os.getenv("BT_API_INTERNAL_URL", f"http://localhost:{_API_PORT}")
    ws_base  = internal.replace("https://", "wss://").replace("http://", "ws://")
    return f"{ws_base}/api/v1/ml/extract/ws"


def _extraction_thread_fn(
    data_types: List[str],
    msg_list: List[Dict],
    done_event: threading.Event,
    error_box: List[str],
    stop_event: threading.Event,
) -> None:
    """Daemon thread: connects to the extraction WebSocket and reads all progress
    messages until 'all_done'/'cancelled' is received or the connection closes.

    stop_event is checked on every receive timeout so the UI Stop button can
    close the connection within ~0.3 s.  msg_list and error_box are shared
    objects — list.append() is effectively atomic under the GIL.
    """
    import asyncio  # noqa: PLC0415

    async def _run() -> None:
        import websockets  # noqa: PLC0415
        url = _extraction_ws_url()
        log.info("Extraction WebSocket client connecting", extra={"url": url})
        # open_timeout=30 because the first document may take a few seconds to load
        # the forensics engine before it can send the first message.
        async with websockets.connect(url, ping_interval=None, open_timeout=30) as ws:
            await ws.send(json.dumps({"data_types": data_types}))
            # Poll in short bursts so the stop_event is checked frequently.
            while True:
                if stop_event.is_set():
                    # Close gracefully — the server will set its own cancel flag
                    # when it detects the WebSocketDisconnect.
                    log.info("Stop requested — closing extraction WebSocket")
                    await ws.close(1001, "Cancelled by user")
                    break
                try:
                    msg_text = await asyncio.wait_for(ws.recv(), timeout=0.3)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    # Connection closed from the server side
                    break
                try:
                    msg = json.loads(msg_text)
                except Exception:
                    continue
                msg_list.append(msg)
                if msg.get("type") in ("all_done", "cancelled"):
                    break

    try:
        asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        error_box.append(str(exc))
        log.error("Extraction WebSocket error", extra={"error": str(exc)})
    finally:
        done_event.set()


def _group_verdict(v: str) -> str:
    """Map a raw forensics verdict string to one of four display groups.

    Used by both the extraction log renderer and the chart builder so that
    verdict grouping is consistent across the UI.
    """
    v_up = v.upper()
    if "ORIGINAL" in v_up and "DERIVED" not in v_up:
        return "ORIGINAL"
    elif "TAMPERED" in v_up:
        return "TAMPERED"
    elif "UNCERTAIN" in v_up:
        return "UNCERTAIN"
    return "OTHER"


def _render_extraction_log(msg_list: List[Dict]) -> None:
    """Render a styled real-time log for the extraction process.

    Shows one line per file processed (with verdict + score), plus banners for
    each step start/done/skip and a final 'all done' summary line.
    """
    lines = []
    for msg in msg_list:
        t     = msg.get("type", "")
        ts    = datetime.now().strftime("%H:%M:%S")
        sname = msg.get("step_name", "")
        # Colour-code by data type: blue for PDFs, purple for images.
        dtype_color = "#3b82f6" if "pdf" in sname.lower() else "#a855f7"

        if t == "step_start":
            lines.append(
                f'<div class="mlt-log-line">'
                f'<span class="mlt-log-ts">{ts}</span>'
                f'<span class="mlt-log-tag" style="color:#f59e0b">'
                f'▶ Step {msg["step_num"]}/{msg["total_steps"]}</span>'
                f'<span class="mlt-log-step">{sname} — <b>{msg["file_count"]}</b> files'
                f' in <code>{msg["folder"]}</code></span>'
                f'</div>'
            )
        elif t == "file_done":
            ok      = msg.get("ok", True)
            verdict = msg.get("verdict", "?")
            score   = msg.get("score", 0)
            # Colour the verdict text by group.
            grp = _group_verdict(verdict)
            v_color = {"ORIGINAL": "#22c55e", "TAMPERED": "#ef4444",
                       "UNCERTAIN": "#f59e0b"}.get(grp, "#94a3b8")
            icon     = "✅" if ok else "❌"
            err_part = (
                f'<span style="color:#ef4444"> — {msg["error"]}</span>'
                if not ok and "error" in msg else ""
            )
            lines.append(
                f'<div class="mlt-log-line">'
                f'<span class="mlt-log-ts">{ts}</span>'
                f'<span class="mlt-log-tag" style="color:{dtype_color}">'
                f'[{msg["index"]}/{msg["total"]}]</span>'
                f'<span class="mlt-log-step">{icon} {msg["file"]}'
                f' <span style="color:{v_color}">{verdict}</span>'
                f' <span style="color:#475569">({score})</span>'
                f'{err_part}</span>'
                f'</div>'
            )
        elif t == "step_done":
            failed_part = f', {msg["rows_failed"]} failed' if msg.get("rows_failed") else ""
            lines.append(
                f'<div class="mlt-log-line">'
                f'<span class="mlt-log-ts">{ts}</span>'
                f'<span class="mlt-log-tag" style="color:#22c55e">✅ Step {msg["step_num"]} done</span>'
                f'<span class="mlt-log-done">{msg["rows_written"]} rows written{failed_part}</span>'
                f'</div>'
            )
        elif t == "step_skip":
            lines.append(
                f'<div class="mlt-log-line">'
                f'<span class="mlt-log-ts">{ts}</span>'
                f'<span class="mlt-log-tag" style="color:#f59e0b">⏭ Step {msg["step_num"]} skipped</span>'
                f'<span class="mlt-log-step">{msg.get("reason", "")}</span>'
                f'</div>'
            )
        elif t == "all_done":
            lines.append(
                f'<div class="mlt-log-line">'
                f'<span class="mlt-log-ts">{ts}</span>'
                f'<span class="mlt-log-done">🎉 Extraction complete — '
                f'{msg.get("total_rows", 0)} rows, {msg.get("total_failed", 0)} failed, '
                f'{msg.get("elapsed_s", 0)}s elapsed</span>'
                f'</div>'
            )
        elif t == "error":
            lines.append(
                f'<div class="mlt-log-line">'
                f'<span class="mlt-log-ts">{ts}</span>'
                f'<span class="mlt-log-err">❌ ERROR: {msg.get("message", "")}</span>'
                f'</div>'
            )

    st.markdown('<div class="mlt-log-box">' + "".join(lines) + "</div>", unsafe_allow_html=True)


def _build_extraction_charts(msg_list: List[Dict], folder_stats: List[Dict]) -> None:
    """Render four matplotlib analytics charts from the completed extraction run.

    Chart A — Input folder sizes: How many files are in each sample folder.
    Chart B — Forensics engine verdict distribution: What % were ORIGINAL vs TAMPERED.
    Chart C — Forgery score histogram: Distribution of raw scores across all files.
    Chart D — Per-folder verdict breakdown: Verdict counts within each extraction step.
    """
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import matplotlib.patches as mpatches  # noqa: PLC0415
        from matplotlib.colors import LinearSegmentedColormap  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        st.warning(f"matplotlib not available for charts: {exc}")
        return

    # Extract all per-file events from the log — these carry verdict + score.
    file_events = [m for m in msg_list if m.get("type") == "file_done"]
    if not file_events:
        st.info("No file events found in the extraction log — re-run extraction to see charts.")
        return

    verdicts = [m.get("verdict", "UNKNOWN") for m in file_events]
    scores   = [float(m.get("score", 0)) for m in file_events]

    # ── Chart A: Input folder sizes ──────────────────────────────────────────
    with st.expander("📁  Chart A — Input sample folder sizes", expanded=True):
        st.markdown(
            "How many files currently sit in each of the 6 sample folders. "
            "**Balanced folder sizes** (similar bar heights) make a more reliable model — "
            "the training algorithm sees roughly equal examples of each class.",
        )
        labels_a = [
            f"{r['label']}\n({r['folder'].replace('_', ' ')})"
            for r in folder_stats
        ]
        counts_a = [r["files"] for r in folder_stats]
        colors_a = ["#22c55e", "#3b82f6", "#ef4444", "#a855f7", "#06b6d4", "#f97316"]

        fig_a, ax_a = plt.subplots(figsize=(9, 3.5), facecolor="#0f172a")
        ax_a.set_facecolor("#1e293b")
        bars_a = ax_a.bar(labels_a, counts_a, color=colors_a[:len(labels_a)],
                          edgecolor="#334155", width=0.55)
        for bar, cnt in zip(bars_a, counts_a):
            ax_a.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                str(cnt), ha="center", va="bottom", color="white",
                fontweight="bold", fontsize=9,
            )
        ax_a.set_ylabel("File count", color="white")
        ax_a.tick_params(colors="white")
        for spine in ax_a.spines.values():
            spine.set_edgecolor("#334155")
        ax_a.grid(axis="y", color="#334155", linewidth=0.6, alpha=0.5)
        ax_a.set_title("Sample Folder Sizes", color="white", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_a, use_container_width=True)
        plt.close(fig_a)

    # ── Chart B: Verdict distribution ────────────────────────────────────────
    with st.expander("🔍  Chart B — Forensics engine verdict distribution"):
        st.markdown(
            "What verdict the forensics engine assigned to each processed file. "
            "Genuine folders should be mostly green; tampered folders mostly red. "
            "Amber 'Uncertain' files sit in the middle ground — the model will learn "
            "to push those into one class or the other.",
        )
        verdict_order  = ["ORIGINAL", "UNCERTAIN", "TAMPERED", "OTHER"]
        verdict_colors = {
            "ORIGINAL": "#22c55e", "UNCERTAIN": "#f59e0b",
            "TAMPERED": "#ef4444", "OTHER": "#64748b",
        }
        grouped: Dict[str, int] = {}
        for v in verdicts:
            g = _group_verdict(v)
            grouped[g] = grouped.get(g, 0) + 1

        v_labels = [v for v in verdict_order if v in grouped]
        v_counts = [grouped[v] for v in v_labels]
        v_colors = [verdict_colors[v] for v in v_labels]
        total_v  = len(verdicts)

        fig_b, ax_b = plt.subplots(figsize=(6, 3), facecolor="#0f172a")
        ax_b.set_facecolor("#1e293b")
        bars_b = ax_b.barh(v_labels, v_counts, color=v_colors, edgecolor="#334155", height=0.5)
        for bar, cnt in zip(bars_b, v_counts):
            ax_b.text(
                bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{cnt}  ({cnt / max(1, total_v):.0%})",
                va="center", color="white", fontsize=9,
            )
        ax_b.set_xlabel("Files", color="white")
        ax_b.tick_params(colors="white")
        ax_b.invert_yaxis()
        for spine in ax_b.spines.values():
            spine.set_edgecolor("#334155")
        ax_b.grid(axis="x", color="#334155", linewidth=0.6, alpha=0.5)
        ax_b.set_title("Forensic Verdict Distribution", color="white", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)

    # ── Chart C: Forgery score histogram ─────────────────────────────────────
    with st.expander("📊  Chart C — Forgery score distribution (0 = clean · 100 = certain tamper)"):
        st.markdown(
            "Each file's raw forensic score plotted as a histogram. "
            "**Green zone (0–40)**: original-looking. **Amber (40–65)**: uncertain. "
            "**Red (65–100)**: likely tampered. Two clearly separated peaks "
            "(one low, one high) mean the model will have an easier time learning "
            "the decision boundary.",
        )
        import numpy as np  # noqa: PLC0415 — already imported above, but keep explicit
        bins_c = np.linspace(0, 100, 21)  # 20 bins of 5 points each

        fig_c, ax_c = plt.subplots(figsize=(7, 3.5), facecolor="#0f172a")
        ax_c.set_facecolor("#1e293b")
        _, bin_edges_c, patches_c = ax_c.hist(scores, bins=bins_c, edgecolor="#334155")
        # Colour each bar according to the score zone it represents.
        for patch, left_edge in zip(patches_c, bin_edges_c[:-1]):
            patch.set_facecolor(
                "#22c55e" if left_edge < 40 else "#f59e0b" if left_edge < 65 else "#ef4444"
            )
        ax_c.set_xlabel("Forgery Score (0–100)", color="white")
        ax_c.set_ylabel("File count", color="white")
        ax_c.tick_params(colors="white")
        for spine in ax_c.spines.values():
            spine.set_edgecolor("#334155")
        ax_c.grid(axis="y", color="#334155", linewidth=0.6, alpha=0.5)
        ax_c.set_title("Forgery Score Distribution", color="white", fontsize=11)
        legend_patches = [
            mpatches.Patch(color="#22c55e", label="0–40 (Original zone)"),
            mpatches.Patch(color="#f59e0b", label="40–65 (Uncertain zone)"),
            mpatches.Patch(color="#ef4444", label="65–100 (Tampered zone)"),
        ]
        ax_c.legend(handles=legend_patches, fontsize=8, facecolor="#0f172a",
                    labelcolor="white", edgecolor="#334155")
        plt.tight_layout()
        st.pyplot(fig_c, use_container_width=True)
        plt.close(fig_c)

    # ── Chart D: Per-folder verdict breakdown ─────────────────────────────────
    with st.expander("🗂️  Chart D — Per-folder verdict breakdown"):
        st.markdown(
            "Grouped bars showing the verdict breakdown *within each extraction step* "
            "(i.e. each sample folder). This reveals which folders produced strong "
            "forensic signals and which produced noisy results worth reviewing.",
        )
        from collections import defaultdict  # noqa: PLC0415
        import numpy as _np  # noqa: PLC0415

        # Build: step_name → {verdict_group: count}
        step_verdicts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for ev in file_events:
            sn = ev.get("step_name", "?")
            vg = _group_verdict(ev.get("verdict", "?"))
            step_verdicts[sn][vg] += 1

        unique_steps   = list(step_verdicts.keys())
        verdict_groups = ["ORIGINAL", "UNCERTAIN", "TAMPERED", "OTHER"]
        v_colors_d     = {
            "ORIGINAL": "#22c55e", "UNCERTAIN": "#f59e0b",
            "TAMPERED": "#ef4444", "OTHER": "#64748b",
        }

        n_steps  = len(unique_steps)
        n_grps   = len(verdict_groups)
        x        = _np.arange(n_steps)
        bar_w    = 0.18

        fig_d, ax_d = plt.subplots(
            figsize=(max(7, n_steps * 1.6), 4), facecolor="#0f172a"
        )
        ax_d.set_facecolor("#1e293b")

        for gi, vg in enumerate(verdict_groups):
            vals   = [step_verdicts[sn].get(vg, 0) for sn in unique_steps]
            offset = (gi - n_grps / 2 + 0.5) * bar_w
            bars_d = ax_d.bar(
                x + offset, vals, width=bar_w,
                label=vg, color=v_colors_d[vg], edgecolor="#334155",
            )
            for bar, val in zip(bars_d, vals):
                if val > 0:
                    ax_d.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        str(val), ha="center", va="bottom", color="white", fontsize=7,
                    )

        ax_d.set_xticks(x)
        short = [s.replace(" images", "\nimages").replace(" PDFs", "\nPDFs") for s in unique_steps]
        ax_d.set_xticklabels(short, fontsize=8, color="white")
        ax_d.set_ylabel("File count", color="white")
        ax_d.tick_params(colors="white")
        for spine in ax_d.spines.values():
            spine.set_edgecolor("#334155")
        ax_d.grid(axis="y", color="#334155", linewidth=0.6, alpha=0.5)
        ax_d.legend(fontsize=8, facecolor="#0f172a", labelcolor="white", edgecolor="#334155")
        ax_d.set_title("Per-Folder Verdict Breakdown", color="white", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)


def _render_sample_folder_browser() -> None:
    """Show a summary table of all 6 sample input folders with file counts and status."""
    st.markdown("#### 📁 Sample Folder Status")
    st.markdown(
        "Add your labelled documents to the folders below, then click **Start Extraction** "
        "to run the full forensics engine on every file and write the training CSVs.",
    )
    folder_stats = _get_sample_folder_stats()

    import pandas as pd  # noqa: PLC0415
    df_folders = pd.DataFrame([
        {
            "Type":        "🖼️ Images" if r["data_type"] == "images" else "📄 PDFs",
            "Folder":      r["folder"],
            "Label":       r["label"],
            "Files":       r["files"],
            "Status": (
                "✅ Ready"   if r["exists"] and r["files"] > 0
                else "⚠️ Empty"  if r["exists"]
                else "❌ Missing"
            ),
            "Description": r["description"],
        }
        for r in folder_stats
    ])
    st.dataframe(
        df_folders, hide_index=True, use_container_width=True,
        column_config={
            "Files":       st.column_config.NumberColumn("Files", width="small"),
            "Status":      st.column_config.TextColumn("Status", width="small"),
            "Folder":      st.column_config.TextColumn("Folder", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
        },
    )
    total_img = sum(r["files"] for r in folder_stats if r["data_type"] == "images")
    total_pdf = sum(r["files"] for r in folder_stats if r["data_type"] == "pdfs")
    st.caption(
        f"Total: **{total_img} image files** across 4 image folders  ·  "
        f"**{total_pdf} PDF files** across 2 PDF folders  ·  "
        f"Root: `fraud_model/sample/`",
    )


def _render_extraction_controls() -> None:
    """Render data-type checkboxes and the Start Extraction button.

    Disables checkboxes when the corresponding sample folders are empty, so the
    user gets immediate feedback about what's ready to extract.
    """
    st.markdown("#### ⚙️ Extraction Configuration")

    folder_stats = _get_sample_folder_stats()
    img_total = sum(r["files"] for r in folder_stats if r["data_type"] == "images")
    pdf_total = sum(r["files"] for r in folder_stats if r["data_type"] == "pdfs")

    ec1, ec2 = st.columns(2)
    with ec1:
        do_images = st.checkbox(
            "🖼️  Extract Image Features",
            value=img_total > 0,
            disabled=img_total == 0,
            help=(
                f"Process all {img_total} images across original, original-derived, "
                "tampered, and tampered-derived folders → writes training_data_image.csv."
            ) if img_total > 0 else "No image files found in the 4 image sample folders.",
        )
    with ec2:
        do_pdfs = st.checkbox(
            "📄  Extract PDF Features",
            value=False,
            disabled=pdf_total == 0,
            help=(
                f"Process all {pdf_total} PDFs across original and tampered folders "
                "→ writes training_data_pdf.csv."
            ) if pdf_total > 0 else "No PDF files found in the 2 PDF sample folders.",
        )

    data_types: List[str] = []
    if do_images and img_total > 0:
        data_types.append("images")
    if do_pdfs and pdf_total > 0:
        data_types.append("pdfs")

    if not data_types:
        st.info(
            "Select at least one data type and ensure the sample folders contain files.",
            icon="ℹ️",
        )
        return

    total_files = (img_total if "images" in data_types else 0) + \
                  (pdf_total if "pdfs"   in data_types else 0)
    # Rough estimate: the forensics engine processes ~1 file per second on average.
    est_min = max(1, total_files // 60)
    st.info(
        f"⏱️  Estimated time: **{est_min}–{est_min + 2} min** for {total_files} files. "
        "Each file passes through the full 11-layer forensics engine. "
        "The page auto-refreshes every 0.5 s so you can watch per-file progress.",
        icon="ℹ️",
    )

    if st.button("🔬  Start Extraction", type="primary", key="mlt_extract_start_btn"):
        _ensure_local_api()

        # Reset any state from a previous run before starting the new one.
        st.session_state["mlt_extract_log"]        = []
        st.session_state["mlt_extract_done"]       = False
        st.session_state["mlt_extract_error"]      = None
        st.session_state["mlt_extract_types"]      = data_types
        st.session_state["mlt_extract_start_time"] = time.time()
        st.session_state["mlt_extracting"]         = True
        st.session_state["mlt_extract_cancelled"]  = False

        done_event = threading.Event()
        stop_event = threading.Event()
        error_box: List[str] = []
        st.session_state["mlt_extract_done_event"] = done_event
        st.session_state["mlt_extract_stop_event"] = stop_event
        st.session_state["mlt_extract_error_box"]  = error_box

        t = threading.Thread(
            target=_extraction_thread_fn,
            args=(data_types, st.session_state["mlt_extract_log"], done_event, error_box, stop_event),
            daemon=True,
        )
        t.start()
        log.info("Extraction thread started", extra={"data_types": data_types})
        st.rerun()


def _render_active_extraction() -> None:
    """Render the live per-file progress log and overall progress bar.

    Called on every page rerun while extraction is running. Auto-reruns every
    0.5 s until the background thread sets done_event.
    """
    msg_list   = st.session_state.get("mlt_extract_log", [])
    done_event = st.session_state.get("mlt_extract_done_event")
    stop_event = st.session_state.get("mlt_extract_stop_event")
    dtypes     = st.session_state.get("mlt_extract_types", [])
    start_time = st.session_state.get("mlt_extract_start_time") or time.time()
    error_box  = st.session_state.get("mlt_extract_error_box", [])

    extraction_complete = done_event is not None and done_event.is_set()
    # Has the user already requested a stop but the thread hasn't finished yet?
    stopping = stop_event is not None and stop_event.is_set() and not extraction_complete
    elapsed = time.time() - start_time
    dtype_label = " + ".join(d.title() for d in dtypes)

    # Header row: title on the left, Stop button on the right.
    hdr_left, hdr_right = st.columns([4, 1])
    with hdr_left:
        if stopping:
            st.markdown(f"### ⏹ Stopping {dtype_label} extraction …")
        else:
            st.markdown(f"### ⏳ Extracting {dtype_label} features …")
    with hdr_right:
        # The stop button is only shown while extraction is actively running and
        # before the user has already clicked it.
        if not stopping and not extraction_complete:
            if st.button("⏹ Stop", key="mlt_extract_stop_btn", type="secondary"):
                if stop_event is not None:
                    stop_event.set()
                    log.info("User requested extraction stop")
                st.rerun()

    st.caption(f"Elapsed: {elapsed:.0f}s  ·  {len(msg_list)} messages received")

    # Compute overall progress pct from step counts + within-step file index.
    step_starts = [m for m in msg_list if m.get("type") == "step_start"]
    file_events = [m for m in msg_list if m.get("type") == "file_done"]
    step_dones  = [m for m in msg_list if m.get("type") == "step_done"]

    if extraction_complete:
        pct = 100
    elif step_starts:
        total_steps = step_starts[-1].get("total_steps", 1)
        completed   = len(step_dones)
        # If we're inside a step, compute fractional progress within it.
        active_step = (
            step_starts[-1] if len(step_starts) > len(step_dones) else None
        )
        if active_step:
            in_step_done  = sum(
                1 for e in file_events
                if e.get("step_num") == active_step.get("step_num")
            )
            step_total = active_step.get("file_count", 1)
            frac       = in_step_done / max(1, step_total)
        else:
            frac = 0.0
        pct = int(((completed + frac) / max(1, total_steps)) * 100)
    else:
        pct = 0

    st.progress(pct / 100, text=f"{pct}% complete")

    if msg_list:
        _render_extraction_log(msg_list)
    else:
        st.info("Connecting to extraction server … this usually takes 2–3 seconds.", icon="🔌")

    if error_box:
        st.error(
            f"WebSocket error — could not connect to the extraction server: {error_box[0]}\n\n"
            "Make sure the BaseTruth API server is running.",
            icon="🔴",
        )

    if extraction_complete:
        # A "cancelled" message in the log means the user stopped the run early.
        was_cancelled = (
            stop_event is not None and stop_event.is_set()
        ) or any(m.get("type") == "cancelled" for m in msg_list)

        all_done = next(
            (m for m in reversed(msg_list) if m.get("type") in ("all_done", "cancelled")),
            {},
        )
        rows_written = all_done.get("total_rows", sum(
            1 for m in msg_list if m.get("type") == "file_done" and m.get("ok")
        ))

        st.session_state["mlt_extracting"]        = False
        st.session_state["mlt_extract_cancelled"] = was_cancelled
        # Mark as done only if it completed normally (not stopped), so the results
        # tab shows a summary; a cancelled run stays actionable.
        st.session_state["mlt_extract_done"] = not was_cancelled

        if was_cancelled:
            st.warning(
                f"⏹  Extraction stopped. **{rows_written} rows** written so far "
                f"({elapsed:.0f}s). You can restart or view partial results.",
                icon="⏹",
            )
        elif not error_box:
            st.success(
                f"✅  Extraction complete! "
                f"{all_done.get('total_rows', 0)} rows written in {all_done.get('elapsed_s', 0)}s.",
                icon="✅",
            )
        time.sleep(0.5)
        st.rerun()
        return

    # Auto-refresh at 0.5 s intervals while extraction is still running.
    time.sleep(0.5)
    st.rerun()


def _render_extraction_results() -> None:
    """Show summary metric cards and all 4 analytics charts after extraction finishes."""
    msg_list = st.session_state.get("mlt_extract_log", [])
    dtypes   = st.session_state.get("mlt_extract_types", [])
    start_t  = st.session_state.get("mlt_extract_start_time") or time.time()
    duration = time.time() - start_t

    all_done     = next((m for m in reversed(msg_list) if m.get("type") == "all_done"), {})
    total_rows   = all_done.get("total_rows", 0)
    total_failed = all_done.get("total_failed", 0)
    elapsed_s    = all_done.get("elapsed_s", round(duration))

    st.success(
        f"✅  Extraction finished in {elapsed_s}s — "
        f"**{total_rows} rows** written"
        + (f", {total_failed} files failed" if total_failed else "")
        + f". The training CSV{'s are' if len(dtypes) > 1 else ' is'} ready for model training.",
        icon="✅",
    )

    # ── Summary metric cards ─────────────────────────────────────────────────
    file_events = [m for m in msg_list if m.get("type") == "file_done"]
    total_ok    = sum(1 for e in file_events if e.get("ok"))
    total_err   = sum(1 for e in file_events if not e.get("ok"))
    avg_score   = (
        sum(e.get("score", 0) for e in file_events) / max(1, len(file_events))
    )

    cards_html = (
        '<div class="mlt-status-grid">'
        + _metric_card("Rows written", str(total_rows),
                       "Across all selected folders", "good" if total_rows > 0 else "bad")
        + _metric_card("Files processed", str(total_ok),
                       f"{total_err} failed" if total_err else "All successful",
                       "good" if total_err == 0 else "warn")
        + _metric_card("Avg forgery score", f"{avg_score:.1f}",
                       "0 = clean, 100 = highly tampered",
                       "good" if avg_score < 40 else "warn")
        + _metric_card("Elapsed", f"{elapsed_s}s",
                       f"≈ {elapsed_s / 60:.1f} min total", "good")
        + "</div>"
    )
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("#### 📈 Extraction Analytics")
    folder_stats = _get_sample_folder_stats()
    _build_extraction_charts(msg_list, folder_stats)


# ─── Main render function ──────────────────────────────────────────────────────

def render() -> None:
    """Entry point called by app.py — renders the full ML Training Pipeline page.

    The page is split into three tabs:
      📦 Data Extraction  — scan sample folders with the forensics engine, watch
                            per-file progress, then explore 4 analytics charts.
      🤖 Model Training   — train XGBoost models from the extracted CSVs, watch
                            live WebSocket progress, review accuracy / F1 / AUC.
      🔍 Signal Reference — plain-English guide to every forensic signal.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(_page_title("🧠", "ML Training Pipeline"), unsafe_allow_html=True)
    st.markdown(
        "Train the fraud-detection AI models, watch live progress, and explore which "
        "forensic signals drive the final verdict for each document class.",
    )

    # ── Initialise session state — both extraction and training keys ─────────
    for key, default in [
        # Training
        ("mlt_training",    False),
        ("mlt_done",        False),
        ("mlt_log",         []),
        ("mlt_results",     {}),
        ("mlt_models",      []),
        ("mlt_error",       None),
        ("mlt_done_event",  None),
        ("mlt_start_time",  None),
        # Extraction
        ("mlt_extracting",         False),
        ("mlt_extract_done",       False),
        ("mlt_extract_log",        []),
        ("mlt_extract_types",      []),
        ("mlt_extract_error",      None),
        ("mlt_extract_done_event", None),
        ("mlt_extract_start_time", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Current model status (shown at the top, above the tabs) ─────────────
    _render_status_section()

    # ── Three-tab layout ─────────────────────────────────────────────────────
    tab_extract, tab_train, tab_signals = st.tabs([
        "📦  Data Extraction",
        "🤖  Model Training",
        "🔍  Signal Reference",
    ])

    # ── Tab 1: Data Extraction ───────────────────────────────────────────────
    with tab_extract:
        st.markdown(
            "Run the full 11-layer forensics engine on every file in the sample folders "
            "to build the labelled training CSVs. "
            "Once extraction is done, switch to the **Model Training** tab to train the models.",
        )
        if st.session_state["mlt_extracting"]:
            # Show live per-file progress — auto-reruns every 0.5 s.
            _render_active_extraction()
        else:
            if st.session_state["mlt_extract_done"] and st.session_state["mlt_extract_log"]:
                # Post-extraction analytics charts.
                _render_extraction_results()
                st.markdown("---")
                if st.button("🔄  Extract again", key="mlt_extract_again"):
                    st.session_state["mlt_extract_done"] = False
                    st.session_state["mlt_extract_log"]  = []
                    st.rerun()
                st.markdown("---")
            # Folder browser + controls always visible unless actively extracting.
            _render_sample_folder_browser()
            st.markdown("---")
            _render_extraction_controls()

    # ── Tab 2: Model Training ────────────────────────────────────────────────
    with tab_train:
        st.markdown(
            "Train the XGBoost fraud-detection models from the CSVs built in the "
            "**Data Extraction** tab. Live metrics and feature-importance charts "
            "appear once training completes.",
        )
        if st.session_state["mlt_training"]:
            # Show live training log — auto-reruns every 0.5 s.
            _render_active_training()
        else:
            if st.session_state["mlt_done"] and st.session_state["mlt_results"]:
                _render_results()
                st.markdown("---")
                if st.button("⬅️  Train again", key="mlt_train_again"):
                    st.session_state["mlt_done"]    = False
                    st.session_state["mlt_results"] = {}
                    st.session_state["mlt_log"]     = []
                    st.rerun()
                st.markdown("---")
            _render_controls()

    # ── Tab 3: Signal Reference ──────────────────────────────────────────────
    with tab_signals:
        img_st  = _get_model_status("image")
        pdf_st  = _get_model_status("pdf")
        live_st = _get_model_status("face_scan_live")

        sig_img_tab, sig_pdf_tab, sig_live_tab = st.tabs([
            f"🖼️ Image Signals ({len(_SIGNAL_GUIDE)})",
            f"📄 PDF Signals ({len(_PDF_SIGNAL_GUIDE)})",
            f"🎥 Face Scan Live Signals (20)",
        ])

        with sig_img_tab:
            from basetruth.analysis.ml_scorer import FEATURE_NAMES as _IMG_FN  # noqa: PLC0415
            # Build the set of feature names the saved image model was trained on.
            # If no model exists yet, pass None so all cards appear active.
            img_active = (
                set(_IMG_FN[: img_st["n_features"]]) if img_st["model_exists"] else None
            )
            _render_signal_guide(
                guide=_SIGNAL_GUIDE,
                active_names=img_active,
                title=f"Understanding the {len(_SIGNAL_GUIDE)} Image Forensic Signals",
                intro=(
                    f"The **image model** is trained on **{len(_SIGNAL_GUIDE)} raw signals** "
                    "extracted from every scanned image document. "
                    "These signals look for ELA compression artefacts, copy-paste cloning, "
                    "colour anomalies, font inconsistencies, and AI-generation patterns. "
                    "Here is what each one looks for, in plain language:"
                ),
            )

        with sig_pdf_tab:
            from basetruth.analysis.ml_scorer_pdf import PDF_FEATURE_NAMES as _PDF_FN  # noqa: PLC0415
            # Build the set of feature names the saved PDF model was trained on.
            # If no model exists yet, pass None so all cards appear active.
            pdf_active = (
                set(_PDF_FN[: pdf_st["n_features"]]) if pdf_st["model_exists"] else None
            )
            _render_signal_guide(
                guide=_PDF_SIGNAL_GUIDE,
                active_names=pdf_active,
                title=f"Understanding the {len(_PDF_SIGNAL_GUIDE)} PDF Forensic Signals",
                intro=(
                    f"The **PDF model** is trained on **{len(_PDF_SIGNAL_GUIDE)} raw signals** "
                    "extracted from every PDF document. "
                    "These signals are completely different from image signals — they examine "
                    "PDF structure, metadata integrity, hidden text, digital signatures, "
                    "incremental update history, and object-level anomalies. "
                    "Here is what each one looks for, in plain language:"
                ),
            )

        with sig_live_tab:
            from basetruth.face_scan.ml_scorer_live import FEATURE_NAMES as _LIVE_FN  # noqa: PLC0415
            live_active = (
                set(_LIVE_FN[: live_st["n_features"]]) if live_st["model_exists"] else None
            )
            _render_signal_guide(
                guide=_LIVE_SIGNAL_GUIDE,
                active_names=live_active,
                title="Understanding the 20 Face Scan Live Signals",
                intro=(
                    "The **Face Scan Live model** is trained on **20 signals** captured during a "
                    "real-time webcam session. These signals measure head-motion smoothness, "
                    "eye micro-movement, replay artefacts, depth cues, and screen-frequency "
                    "patterns to distinguish a genuine live person from a replay or virtual-camera attack."
                ),
            )


def _render_controls() -> None:
    """Render the model selection checkboxes and Start Training button."""
    st.markdown("#### ⚙️ Training Configuration")

    img_avail = _IMAGE_CSV.exists()
    pdf_avail = _PDF_CSV.exists()

    live_avail = _LIVE_CSV.exists()

    col1, col2, col3 = st.columns(3)
    with col1:
        train_image = st.checkbox(
            "🖼️  Train Image Model",
            value=True,
            disabled=not img_avail,
            help="Uses fraud_model/data/training_data_image.csv — 4-class multiclass (ORIGINAL / ORIGINAL-DERIVED / TAMPERED / TAMPERED-DERIVED)" if img_avail else "training_data_image.csv not found",
        )
    with col2:
        train_pdf = st.checkbox(
            "📄  Train PDF Model",
            value=False,
            disabled=not pdf_avail,
            help="Uses fraud_model/data/training_data_pdf.csv — binary (genuine / tampered)" if pdf_avail else "training_data_pdf.csv not found",
        )
    with col3:
        train_live = st.checkbox(
            "🎥  Train Face Scan Live Model",
            value=False,
            disabled=not live_avail,
            help="Uses fraud_model/data/training_data_face_scan_live.csv — binary (GENUINE / SPOOF). Run scripts/collect_live_scan_samples.py first to build this CSV." if live_avail else "training_data_face_scan_live.csv not found — run scripts/collect_live_scan_samples.py first",
        )

    models_to_train = []
    if train_image and img_avail:
        models_to_train.append("image")
    if train_pdf and pdf_avail:
        models_to_train.append("pdf")
    if train_live and live_avail:
        models_to_train.append("face_scan_live")

    if not models_to_train:
        st.info(
            "Select at least one model to train and make sure the training CSV exists.",
            icon="ℹ️",
        )
        return

    # Warn how long training will take based on sample count
    total_samples = 0
    for mt in models_to_train:
        _mt_status = _get_model_status(mt)
        total_samples += _mt_status.get("n_samples", 0)

    est_minutes = max(1, total_samples // 150)

    st.info(
        f"⏱️  Estimated training time: **{est_minutes}–{est_minutes + 1} minute(s)** "
        f"for {total_samples} samples across {len(models_to_train)} model(s). "
        "The page will auto-refresh every 0.5 s so you can watch the live log.",
        icon="ℹ️",
    )

    if st.button("🚀  Start Training", type="primary", key="mlt_start_btn"):
        # Make sure the FastAPI server is running so the WebSocket is available
        _ensure_local_api()

        # Reset state from any previous run
        st.session_state["mlt_log"]        = []
        st.session_state["mlt_results"]    = {}
        st.session_state["mlt_done"]       = False
        st.session_state["mlt_error"]      = None
        st.session_state["mlt_models"]     = models_to_train
        st.session_state["mlt_start_time"] = time.time()
        st.session_state["mlt_training"]   = True

        done_event = threading.Event()
        error_box: List[str] = []
        st.session_state["mlt_done_event"] = done_event
        st.session_state["mlt_error_box"]  = error_box

        t = threading.Thread(
            target=_training_thread_fn,
            args=(models_to_train, st.session_state["mlt_log"], done_event, error_box),
            daemon=True,
        )
        t.start()
        log.info("Training thread started", extra={"models": models_to_train})
        st.rerun()


def _render_active_training() -> None:
    """Render the live training log and progress bar."""
    import streamlit as _st  # noqa: PLC0415 — local alias to avoid shadowing

    msg_list   = _st.session_state.get("mlt_log", [])
    done_event = _st.session_state.get("mlt_done_event")
    models     = _st.session_state.get("mlt_models", [])
    start_time = _st.session_state.get("mlt_start_time") or time.time()
    error_box  = _st.session_state.get("mlt_error_box", [])

    # Check whether the background thread has finished
    training_complete = done_event is not None and done_event.is_set()

    elapsed = time.time() - start_time
    model_label = " + ".join(m.title() for m in models)

    _st.markdown(f"### ⏳ Training {model_label} Model{'s' if len(models) > 1 else ''}...")
    _st.caption(f"Elapsed: {elapsed:.0f}s  |  Received {len(msg_list)} log messages")

    # Overall progress: compute from the latest pct value in the log
    log_items = [m for m in msg_list if m.get("type") == "log"]
    current_pct = log_items[-1].get("pct", 0) if log_items else 0
    if training_complete:
        current_pct = 100

    _st.progress(current_pct / 100, text=f"{current_pct}% complete")

    # Render the live log feed
    if msg_list:
        _render_log(msg_list)
    else:
        _st.info("Connecting to training server... this usually takes 2–3 seconds.", icon="🔌")

    # Handle errors reported by the thread
    if error_box:
        _st.error(
            f"WebSocket error — could not connect to the training server: {error_box[0]}\n\n"
            "Make sure the BaseTruth API server is running (it starts automatically for local installs).",
            icon="🔴",
        )

    # When done, collect results from the log and transition to the results view
    if training_complete:
        results: Dict[str, Any] = {}
        for msg in msg_list:
            if msg.get("type") == "done":
                results[msg["model"]] = msg.get("metrics", {})

        _st.session_state["mlt_training"] = False
        _st.session_state["mlt_done"]     = True
        _st.session_state["mlt_results"]  = results

        if not error_box:
            _st.success("✅  Training complete! Scroll down to see the results.", icon="✅")
        time.sleep(0.5)
        _st.rerun()
        return

    # Auto-refresh every 0.5 s while training is in progress
    time.sleep(0.5)
    _st.rerun()


def _render_results() -> None:
    """Render metrics cards and analytics charts for all trained models."""
    import streamlit as _st  # noqa: PLC0415

    results  = _st.session_state.get("mlt_results", {})
    models   = _st.session_state.get("mlt_models", list(results.keys()))
    start_t  = _st.session_state.get("mlt_start_time") or time.time()
    duration = time.time() - start_t

    _st.success(
        f"✅  Training finished in {duration / 60:.1f} minute(s). "
        f"Model{'s' if len(models) > 1 else ''}: {', '.join(m.title() for m in models)}.",
        icon="✅",
    )

    for model_type in models:
        metrics = results.get(model_type, {})
        if not metrics:
            continue

        label = "Image" if model_type == "image" else "PDF"
        _st.markdown(f"---\n#### 🎯 {label} Model — Performance Summary")
        _render_metrics_cards(metrics, model_type)

        _st.markdown(f"#### 📈 {label} Model — Visual Analytics")
        if model_type == "image":
            _build_image_charts(metrics)
        else:
            _build_pdf_charts(metrics)
