import argparse
import sqlite3
from datetime import datetime
from typing import Optional


def iso_to_dt(s: str) -> datetime:
    # ts stored as ISO string like 2026-01-01T00:00:05
    return datetime.fromisoformat(str(s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--model-version", default="logreg_v1")
    parser.add_argument(
        "--min-stable", type=int, default=3,
        help="Require K consecutive same predictions to confirm a location (reduces jitter)."
    )
    parser.add_argument(
        "--dwell-seconds", type=int, default=10,
        help="Create DWELL if continuous time in same loc >= this many seconds."
    )
    parser.add_argument(
        "--clear-existing", action="store_true",
        help="Delete existing ARRIVAL/DEPARTURE/DWELL events before writing new ones."
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON;")

    if args.clear_existing:
        conn.execute("DELETE FROM events WHERE event_type IN ('ARRIVAL','DEPARTURE','DWELL');")
        conn.commit()

    rows = conn.execute(
        """
        SELECT s.scan_id, s.tag_id, s.ts, p.pred_loc, p.pred_conf
        FROM scans s
        JOIN loc_predictions p ON p.scan_id = s.scan_id
        WHERE p.model_version = ?
        ORDER BY s.tag_id, s.ts;
        """,
        (args.model_version,),
    ).fetchall()

    if not rows:
        raise RuntimeError("No predictions found. Did you run training and write loc_predictions?")

    events_to_insert = []

    def emit_event(tag_id, loc, event_type, start_ts, end_ts, conf):
        events_to_insert.append((str(tag_id), str(loc), str(event_type), str(start_ts),
                                 None if end_ts is None else str(end_ts), float(conf)))

    def close_segment(tag_id: str, loc: str, start_ts: str, end_ts: str,
                      conf_sum: float, n: int):
        """
        End a stable segment at end_ts:
          - emit DWELL if duration >= dwell_seconds
          - emit DEPARTURE (always)
        """
        if loc is None or start_ts is None or end_ts is None:
            return
        dur = (iso_to_dt(end_ts) - iso_to_dt(start_ts)).total_seconds()
        avg_conf = conf_sum / max(n, 1)
        if dur >= args.dwell_seconds:
            emit_event(tag_id, loc, "DWELL", start_ts, end_ts, avg_conf)
        emit_event(tag_id, loc, "DEPARTURE", end_ts, None, avg_conf)

    # Per-tag state
    current_tag: Optional[str] = None

    # Stability window (K consecutive same predicted LOC)
    buffer_loc: Optional[str] = None
    buffer_count: int = 0

    # Current stable segment
    stable_loc: Optional[str] = None
    stable_start_ts: Optional[str] = None
    stable_conf_sum: float = 0.0
    stable_n: int = 0

    last_ts_for_current_tag: Optional[str] = None

    for scan_id, tag_id, ts, pred_loc, pred_conf in rows:
        tag_id = str(tag_id)
        ts = str(ts)
        pred_loc = str(pred_loc)
        pred_conf = float(pred_conf) if pred_conf is not None else 0.0

        # Tag boundary
        if current_tag is None:
            current_tag = tag_id

        if tag_id != current_tag:
            # Flush the previous tag using its own last timestamp
            if stable_loc is not None and stable_start_ts is not None and last_ts_for_current_tag is not None:
                close_segment(current_tag, stable_loc, stable_start_ts, last_ts_for_current_tag,
                              stable_conf_sum, stable_n)

            # Reset for new tag
            current_tag = tag_id
            buffer_loc = None
            buffer_count = 0
            stable_loc = None
            stable_start_ts = None
            stable_conf_sum = 0.0
            stable_n = 0
            last_ts_for_current_tag = None

        last_ts_for_current_tag = ts

        # --- Stability confirmation window ---
        if buffer_loc is None or pred_loc != buffer_loc:
            buffer_loc = pred_loc
            buffer_count = 1
        else:
            buffer_count += 1

        if buffer_count < args.min_stable:
            continue

        # Confirmed stable location (buffer_loc)
        if stable_loc is None:
            # First stable segment for this tag
            stable_loc = buffer_loc
            stable_start_ts = ts
            stable_conf_sum = pred_conf
            stable_n = 1
            emit_event(tag_id, stable_loc, "ARRIVAL", ts, None, pred_conf)
            continue

        if buffer_loc == stable_loc:
            # still in same stable segment
            stable_conf_sum += pred_conf
            stable_n += 1
            continue

        # stable location changed: close previous, start new
        prev_loc = stable_loc
        prev_start = stable_start_ts
        prev_end = ts  # the moment we confirm a new stable loc

        close_segment(tag_id, prev_loc, prev_start, prev_end, stable_conf_sum, stable_n)

        stable_loc = buffer_loc
        stable_start_ts = ts
        stable_conf_sum = pred_conf
        stable_n = 1
        emit_event(tag_id, stable_loc, "ARRIVAL", ts, None, pred_conf)

    # Flush the last tag at end-of-file
    if current_tag is not None and stable_loc is not None and stable_start_ts is not None and last_ts_for_current_tag is not None:
        close_segment(current_tag, stable_loc, stable_start_ts, last_ts_for_current_tag,
                      stable_conf_sum, stable_n)

    conn.executemany(
        """
        INSERT INTO events(tag_id, loc, event_type, start_ts, end_ts, confidence)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        events_to_insert,
    )
    conn.commit()

    print(f"Inserted {len(events_to_insert)} events (ARRIVAL/DEPARTURE/DWELL) using model_version={args.model_version}")
    conn.close()


if __name__ == "__main__":
    main()
