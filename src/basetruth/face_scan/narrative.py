"""Face Scan narrative generator.

Converts a completed Face Scan result dict (static or live) into a plain-English
honest_review written by Gemma4. Falls back gracefully to the rule-based text that
was already in the result when Ollama is offline or returns an empty response.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from basetruth.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt — tells Gemma4 its role and output constraints
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a face authenticity analyst working inside a KYC fraud detection platform called BaseTruth.
An automated system has already analysed a face scan and produced a machine verdict (GENUINE, SUSPICIOUS, DEEPFAKE, or INCONCLUSIVE).
Your job is to translate that verdict and its supporting evidence into a short, plain-English summary that a non-technical compliance officer or HR reviewer can read and act on immediately.

Rules:
- Write exactly 2 to 8 sentences. No bullet points, no numbered lists, no markdown.
- Use plain, everyday language. Avoid technical terms like "Laplacian", "EAR", "Hamming distance", "pHash".
- State the verdict clearly in the first sentence.
- Mention the one or two most important signals that drove the verdict.
- End with a clear action recommendation: approve, escalate for manual review, or reject.
- Do not invent facts. Only use the data provided.
- Do not add any preamble like "Here is the summary:" — output only the review paragraph itself.
"""

# ---------------------------------------------------------------------------
# User prompt template — filled from the result dict
# ---------------------------------------------------------------------------

_USER_PROMPT_TEMPLATE = """\
Face scan result to explain:

Mode: {mode}
Verdict: {verdict}
Risk score: {risk}/100  (0 = safe, 100 = definitely fake)
Confidence: {confidence}/100  (how much to trust this verdict)

Key findings:
{findings}

Evidence bullets from the automated system:
{evidence}

Write the plain-English review now.
"""


def _build_findings(result: Dict[str, Any]) -> str:
    """Extract the most operator-relevant signals into a short prose brief.

    We deliberately avoid dumping raw JSON — a small focused summary gives the
    LLM a better signal-to-noise ratio and keeps token usage low so even the
    smallest gemma4:e2b model produces a useful answer.
    """
    checks = result.get("checks", {})
    mode = result.get("mode", "static")
    lines: list[str] = []

    # --- Replay (live only) ---
    replay = checks.get("replay_heuristics", {})
    if replay:
        score = replay.get("score_0_100", 0.0)
        repeat = replay.get("repeat_frame_score", 0.0)
        if score >= 60.0:
            lines.append(f"Replay detection: HIGH risk ({score:.0f}/100). Many video frames appeared to be identical repeats (repeat_frame_score={repeat:.0f}), which is a strong sign of a pre-recorded video attack.")
        elif score >= 30.0:
            lines.append(f"Replay detection: MODERATE risk ({score:.0f}/100). Some repeated frame patterns detected.")
        else:
            lines.append(f"Replay detection: LOW risk ({score:.0f}/100). Frames varied naturally, consistent with a live person.")

    # --- Temporal consistency (live only) ---
    temporal = checks.get("temporal_consistency", {})
    if temporal:
        score = temporal.get("score_0_100", 0.0)
        if score >= 50.0:
            lines.append(f"Head motion consistency: HIGH risk ({score:.0f}/100). Movements were jerky or erratic, inconsistent with a real person moving in front of a camera.")
        elif score >= 30.0:
            lines.append(f"Head motion consistency: MODERATE risk ({score:.0f}/100). Some unusual movement patterns detected.")
        else:
            lines.append(f"Head motion consistency: LOW risk ({score:.0f}/100). Movements looked smooth and natural.")

    # --- Photo authenticity (static only) ---
    photo_auth = checks.get("photo_authenticity", {})
    if photo_auth:
        pres = photo_auth.get("presentation_risk_0_100", 0.0)
        synth = photo_auth.get("synthetic_risk_0_100", 0.0)
        if pres >= 50.0:
            lines.append(f"Photo authenticity: HIGH presentation risk ({pres:.0f}/100). The image shows signs of being a photo of a screen or printed photo rather than a direct camera capture.")
        elif pres >= 25.0:
            lines.append(f"Photo authenticity: MODERATE presentation risk ({pres:.0f}/100). Some re-photograph or screen-capture signals present.")
        else:
            lines.append(f"Photo authenticity: LOW presentation risk ({pres:.0f}/100). No strong print-photo or screen-capture artefacts found.")
        if synth >= 40.0:
            lines.append(f"Synthetic/manipulation signals: HIGH ({synth:.0f}/100). Face geometry shows elevated asymmetry, suggesting possible manipulation.")

    # --- Quality ---
    quality = checks.get("quality_assessment", {})
    if quality:
        blur = quality.get("blur_risk_0_100", 0.0)
        brightness = quality.get("brightness_risk_0_100", 0.0)
        face_size = quality.get("face_size_risk_0_100", 0.0)
        issues: list[str] = []
        if blur >= 50.0:
            issues.append("blurry image")
        if brightness >= 50.0:
            issues.append("poor lighting")
        if face_size >= 50.0:
            issues.append("face too small in frame")
        if issues:
            lines.append(f"Image quality issues: {', '.join(issues)}. This reduces confidence in the result.")
        else:
            lines.append("Image quality: acceptable sharpness, lighting, and face size.")

    # --- Active liveness (live only) ---
    liveness = checks.get("active_liveness", {})
    if liveness:
        passed = liveness.get("passed", False)
        challenges = liveness.get("completed_challenges", [])
        if passed:
            lines.append(f"All liveness challenges completed: {', '.join(challenges)}.")
        else:
            lines.append("The person did not complete all required liveness challenges.")

    # --- Depth & screen (live, brief) ---
    depth = checks.get("depth_consistency", {})
    if depth and depth.get("score_0_100", 0.0) >= 50.0:
        lines.append(f"3D depth check: HIGH risk ({depth['score_0_100']:.0f}/100). Eye separation did not behave like a real 3D face during head turns, suggesting a flat photo or screen source.")

    screen = checks.get("screen_frequency", {})
    if screen and screen.get("score_0_100", 0.0) >= 50.0:
        lines.append(f"Screen frequency check: HIGH risk ({screen['score_0_100']:.0f}/100). The image shows patterns consistent with being filmed from a digital screen.")

    if not lines:
        lines.append("No specific check findings available.")

    return "\n".join(f"- {line}" for line in lines)


def generate_face_scan_narrative(result: Dict[str, Any]) -> Tuple[str, str]:
    """Call Gemma4 to produce a plain-English honest_review for a face scan result.

    This function takes the completed result dict (from either the static or live
    scan path), builds a compact text brief summarising the key signals, and asks
    the LLM to translate it into 2-4 sentences an operator can read and act on.

    Returns a tuple of (review_text, source) where:
    - review_text: the LLM-written paragraph (or the original rule-based text on fallback)
    - source: "gemma4 (model_name)" on success, "rule_based" on fallback

    This function never raises — any LLM failure returns the existing honest_review
    so the face scan result is always complete regardless of Ollama availability.
    """
    fallback_text: str = result.get("honest_review", "")

    try:
        # Lazy import so the module loads without Ollama installed
        from basetruth.integrations.ollama import _route_vlm_chat  # type: ignore[attr-defined]

        mode = result.get("mode", "static")
        verdict = result.get("verdict", "UNKNOWN")
        risk = result.get("risk_score_0_100", 0.0)
        confidence = result.get("confidence_0_100", 0.0)
        evidence_bullets = result.get("evidence", [])
        evidence_text = "\n".join(f"- {b}" for b in evidence_bullets) if evidence_bullets else "- No evidence bullets available."

        findings = _build_findings(result)

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            mode=mode,
            verdict=verdict,
            risk=f"{risk:.1f}",
            confidence=f"{confidence:.1f}",
            findings=findings,
            evidence=evidence_text,
        )

        # Text-only call — the face scan result is already in text form, no image needed.
        # Use feature="face_scan" so operators can route it to a faster model if desired.
        content, engine, model, _ = _route_vlm_chat(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_bytes_list=[],
            feature="face_scan",
        )

        if content and content.strip():
            log.info(
                "generate_face_scan_narrative: narrative generated — engine=%s model=%s chars=%d",
                engine, model, len(content),
            )
            return content.strip(), f"gemma4 ({model})"

        log.warning(
            "generate_face_scan_narrative: LLM returned empty response — using rule-based fallback"
        )

    except Exception as exc:
        log.warning(
            "generate_face_scan_narrative: LLM call failed (%s) — using rule-based fallback", exc
        )

    return fallback_text, "rule_based"
