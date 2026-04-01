"""Log Analyzer page."""
from __future__ import annotations

import json

import streamlit as st

from basetruth.ui.components import _LOGGER_OK, _log_path


def _page_logs() -> None:
    import pandas as pd  # noqa: PLC0415
    from datetime import datetime as _dt  # noqa: PLC0415

    st.markdown(
        """
        <style>
        .log-header { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
        .log-header h1 { margin:0; font-size:1.8rem; font-weight:700; }
        .metric-row { display:flex; gap:14px; margin:16px 0 20px 0; }
        .metric-card {
            flex:1; padding:18px 20px; border-radius:14px;
            display:flex; flex-direction:column; gap:4px;
            box-shadow: 0 2px 12px rgba(0,0,0,.08);
            transition: transform .15s ease, box-shadow .15s ease;
        }
        .metric-card:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,0,0,.12); }
        .metric-card .mc-value { font-size:1.9rem; font-weight:800; line-height:1.1; }
        .metric-card .mc-label { font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; opacity:.75; font-weight:600; }
        .mc-total   { background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); color:#3730a3; }
        .mc-error   { background: linear-gradient(135deg, #fee2e2 0%, #fca5a5 100%); color:#991b1b; }
        .mc-warn    { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); color:#92400e; }
        .mc-info    { background: linear-gradient(135deg, #d1fae5 0%, #6ee7b7 100%); color:#065f46; }
        .quick-filters { display:flex; gap:8px; flex-wrap:wrap; margin:2px 0 14px 0; }
        .log-detail-card {
            background: #f8fafc; border:1px solid #e2e8f0; border-radius:12px;
            padding:16px 20px; margin-top:10px;
        }
        .log-tail-row {
            padding:7px 14px; border-radius:8px; margin-bottom:4px; font-size:.82rem;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            border-left:4px solid transparent;
        }
        .log-tail-error   { background:#fff1f2; border-left-color:#ef4444; color:#991b1b; }
        .log-tail-warning { background:#fffbeb; border-left-color:#f59e0b; color:#92400e; }
        .log-tail-info    { background:#f0fdf4; border-left-color:#22c55e; color:#166534; }
        .log-tail-debug   { background:#f8fafc; border-left-color:#94a3b8; color:#64748b; }
        .module-chip {
            display:inline-block; padding:3px 10px; border-radius:20px; font-size:.72rem;
            font-weight:600; margin:2px; background:#e2e8f0; color:#334155;
        }
        .module-chip-hot { background:#fecaca; color:#991b1b; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _hdr_l, _hdr_r, _hdr_clr = st.columns([5, 1, 1])
    _hdr_l.markdown(
        '<div class="log-header"><h1>📋 Log Analyzer</h1></div>',
        unsafe_allow_html=True,
    )
    if _hdr_r.button("🔄 Refresh", use_container_width=True, key="log_refresh"):
        st.rerun()

    lp = _log_path() if _LOGGER_OK else None
    if lp is None or not lp.exists():
        st.info(
            "No log file found yet. Run some scans and the log file will appear here.\n\n"
            f"Expected location: `{lp}`"
        )
        return

    if _hdr_clr.button("🗑 Clear", use_container_width=True, key="log_clear"):
        try:
            lp.write_text("", encoding="utf-8")
            st.success("Log file cleared.")
            st.rerun()
        except Exception as _clr_exc:  # noqa: BLE001
            st.error(f"Could not clear log file: {_clr_exc}")

    raw_lines: list = []
    try:
        with open(lp, "r", encoding="utf-8") as fh:
            for _line in fh:
                _line = _line.strip()
                if _line:
                    raw_lines.append(_line)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read log file: {exc}")
        return

    records: list = []
    for _line in raw_lines:
        try:
            records.append(json.loads(_line))
        except json.JSONDecodeError:
            records.append(
                {
                    "ts": "",
                    "level": "RAW",
                    "msg": _line,
                    "module": "",
                    "func": "",
                    "logger": "",
                }
            )

    if not records:
        st.info("Log file exists but is empty. Run some scans first.")
        return

    df = pd.DataFrame(records)
    for col in ["ts", "level", "logger", "module", "func", "line", "msg"]:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")

    _n_total = len(df)
    _n_err = int((df["level"] == "ERROR").sum())
    _n_warn = int((df["level"] == "WARNING").sum())
    _n_info = int(((df["level"] == "INFO") | (df["level"] == "DEBUG")).sum())

    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-card mc-total">
            <span class="mc-value">{_n_total:,}</span>
            <span class="mc-label">Total Entries</span>
          </div>
          <div class="metric-card mc-error">
            <span class="mc-value">{_n_err:,}</span>
            <span class="mc-label">Errors</span>
          </div>
          <div class="metric-card mc-warn">
            <span class="mc-value">{_n_warn:,}</span>
            <span class="mc-label">Warnings</span>
          </div>
          <div class="metric-card mc-info">
            <span class="mc-value">{_n_info:,}</span>
            <span class="mc-label">Info + Debug</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _chart_l, _chart_r = st.columns([2, 1])
    with _chart_l:
        st.markdown("##### 📈 Log Volume Timeline")
        if df["ts"].str.len().max() > 10:
            try:
                _ts_series = pd.to_datetime(df["ts"], errors="coerce")
                _ts_valid = _ts_series.dropna()
                if len(_ts_valid) > 2:
                    _timeline_df = (
                        _ts_valid.dt.floor("min")
                        .value_counts()
                        .sort_index()
                        .rename("count")
                        .reset_index()
                    )
                    _timeline_df.columns = ["time", "count"]
                    st.area_chart(
                        _timeline_df.set_index("time"),
                        height=220,
                        use_container_width=True,
                    )
                else:
                    st.caption(
                        "Not enough timestamped entries for a timeline chart."
                    )
            except Exception:  # noqa: BLE001
                st.caption("Could not parse timestamps for timeline.")
        else:
            st.caption("No timestamps available.")

    with _chart_r:
        st.markdown("##### 🎯 Level Distribution")
        _level_counts = df["level"].value_counts()
        if len(_level_counts) > 0:
            _lc_df = _level_counts.reset_index()
            _lc_df.columns = ["Level", "Count"]
            st.bar_chart(
                _lc_df.set_index("Level"), height=220, use_container_width=True
            )

    st.markdown("##### 🧩 Module Activity")
    _top_modules = df["module"].value_counts().head(10)
    _chips_html = ""
    for _mod, _cnt in _top_modules.items():
        if not _mod:
            continue
        _mod_errors = int((df[df["module"] == _mod]["level"] == "ERROR").sum())
        _cls = "module-chip-hot" if _mod_errors > 0 else ""
        _err_tag = f" · {_mod_errors} err" if _mod_errors else ""
        _chips_html += (
            f'<span class="module-chip {_cls}">'
            f"{_mod} ({_cnt}{_err_tag})</span>"
        )
    if _chips_html:
        st.markdown(
            f'<div style="margin:8px 0 16px 0;">{_chips_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No module data available.")

    st.divider()
    st.markdown("##### 🔍 Filter Logs")
    _qf_cols = st.columns(7)
    if _qf_cols[0].button("🔴 Errors Only", use_container_width=True, key="qf_err"):
        st.session_state["log_level_filter_v2"] = "ERROR"
    if _qf_cols[1].button("🟡 Warnings", use_container_width=True, key="qf_warn"):
        st.session_state["log_level_filter_v2"] = "WARNING"
    if _qf_cols[2].button("🟢 Info", use_container_width=True, key="qf_info"):
        st.session_state["log_level_filter_v2"] = "INFO"
    if _qf_cols[3].button("⚪ Debug", use_container_width=True, key="qf_debug"):
        st.session_state["log_level_filter_v2"] = "DEBUG"
    if _qf_cols[4].button(
        "📋 All Levels", use_container_width=True, key="qf_all"
    ):
        st.session_state["log_level_filter_v2"] = "ALL"

    _active_level = st.session_state.get("log_level_filter_v2", "ALL")
    _f1, _f2, _f3 = st.columns([1.5, 2, 3])
    level_opts = ["ALL"] + sorted(df["level"].unique().tolist())
    _default_idx = (
        level_opts.index(_active_level) if _active_level in level_opts else 0
    )
    module_opts = ["ALL"] + sorted([m for m in df["module"].unique() if m])
    chosen_level = _f1.selectbox(
        "Level", level_opts, index=_default_idx, key="log_level_sel_v2"
    )
    chosen_module = _f2.selectbox(
        "Module", module_opts, key="log_module_sel_v2"
    )
    search_text = _f3.text_input(
        "Search messages", placeholder="keyword…", key="log_search_v2"
    )

    view = df.copy()
    if chosen_level != "ALL":
        view = view[view["level"] == chosen_level]
    if chosen_module != "ALL":
        view = view[view["module"] == chosen_module]
    if search_text:
        view = view[view["msg"].str.contains(search_text, case=False, na=False)]

    st.caption(f"Showing **{len(view):,}** of {len(df):,} entries")

    _LEVEL_COLORS: dict = {
        "ERROR": "background-color:#fee2e2;color:#991b1b;font-weight:700",
        "WARNING": "background-color:#fef9c3;color:#854d0e;font-weight:600",
        "INFO": "background-color:#f0fdf4;color:#166534",
        "DEBUG": "background-color:#f1f5f9;color:#64748b",
    }

    def _style_level(val: str) -> str:
        return _LEVEL_COLORS.get(val, "")

    display_cols = [
        c for c in ["ts", "level", "logger", "func", "msg"] if c in view.columns
    ]
    rename_map = {
        "ts": "Timestamp",
        "level": "Level",
        "logger": "Source",
        "func": "Function",
        "msg": "Message",
    }

    if len(view) > 0:
        styled = (
            view[display_cols]
            .rename(columns=rename_map)
            .style.map(_style_level, subset=["Level"])
        )
        st.dataframe(
            styled, hide_index=True, use_container_width=True, height=480
        )
    else:
        st.info("No log entries match your filters.")

    st.divider()
    st.markdown("##### 🔴 Live Tail — Latest Entries")
    _tail = view.tail(15).iloc[::-1]
    _tail_html = ""
    for _, row in _tail.iterrows():
        _lvl = str(row.get("level", "")).upper()
        _cls = {
            "ERROR": "log-tail-error",
            "WARNING": "log-tail-warning",
            "INFO": "log-tail-info",
            "DEBUG": "log-tail-debug",
        }.get(_lvl, "log-tail-debug")
        _ts_short = (
            str(row.get("ts", ""))[-8:]
            if len(str(row.get("ts", ""))) > 8
            else str(row.get("ts", ""))
        )
        _mod = row.get("logger", "") or row.get("module", "")
        _msg = str(row.get("msg", ""))[:200]
        _tail_html += (
            f'<div class="log-tail-row {_cls}">'
            f"<strong>[{_lvl}]</strong> "
            f'<span style="opacity:.6">{_ts_short}</span> '
            f'<span style="color:#6366f1;font-weight:600">{_mod}</span> '
            f"— {_msg}"
            f"</div>"
        )
    if _tail_html:
        st.markdown(_tail_html, unsafe_allow_html=True)
    else:
        st.caption("No entries to display.")

    st.divider()
    st.markdown("##### 🔎 JSON Inspector")
    st.caption("Select a log entry to view its full structured payload.")
    if len(view) > 0:
        _max_idx = len(view) - 1
        _sel = st.slider(
            "Entry (most recent = 0)",
            min_value=0,
            max_value=_max_idx,
            value=min(0, _max_idx),
            key="log_json_slider",
        )
        _record_idx = (
            view.index[len(view) - 1 - _sel] if _sel <= _max_idx else view.index[0]
        )
        _chosen_record = records[_record_idx]
        _c1, _c2 = st.columns([1, 3])
        with _c1:
            st.markdown(
                f"""
                **Level:** `{_chosen_record.get('level', '?')}`  
                **Module:** `{_chosen_record.get('logger', _chosen_record.get('module', '?'))}`  
                **Function:** `{_chosen_record.get('func', '?')}`  
                **Line:** `{_chosen_record.get('line', '?')}`  
                """,
            )
        with _c2:
            st.json(_chosen_record, expanded=True)

    st.divider()
    _dl1, _dl2, _ = st.columns([1, 1, 4])
    with open(lp, "rb") as fh:
        _dl1.download_button(
            "⬇ Download JSONL",
            data=fh.read(),
            file_name="basetruth.jsonl",
            mime="application/x-ndjson",
            key="log_dl_jsonl",
            use_container_width=True,
        )
    if len(view) > 0:
        _csv_data = view[display_cols].to_csv(index=False)
        _dl2.download_button(
            "⬇ Download CSV",
            data=_csv_data,
            file_name="basetruth_logs_filtered.csv",
            mime="text/csv",
            key="log_dl_csv",
            use_container_width=True,
        )
