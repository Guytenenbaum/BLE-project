import argparse
import sqlite3
from datetime import datetime

def iso_to_dt(s: str) -> datetime:
    # ts stored as ISO string like 2026-01-01T00:00:05
    return datetime.fromisoformat(s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--model-version", default="logreg_v1")
    parser.add_argument("--min-stable", type=int, default=3,
                        help="Require K consecutive same predictions to confirm a zone (reduces jitter).")
    parser.add_argument("--dwell-seconds", type=int, default=60,
                        help="Create DWELL if continuous time in same loc >= this many seconds.")
    parser.add_argument("--clear-existing", action="store_true",
                        help="Delete existing ARRIVAL/DEPARTURE/DWELL events before writing new ones.")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON;")

    if args.clear_existing:
        conn.execute("DELETE FROM events WHERE event_type IN ('ARRIVAL','DEPARTURE','DWELL');")
        conn.commit()

    # Get prediction stream ordered by time
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

    # Group by tag_id
    events_to_insert = []
    current_tag = None
    buffer_loc = None
    buffer_count = 0

    stable_loc = None
    stable_start_ts = None
    stable_conf_sum = 0.0
    stable_n = 0

    def emit_event(tag_id, loc, event_type, start_ts, end_ts, conf):
        events_to_insert.append((tag_id, loc, event_type, start_ts, end_ts, conf))

    # Iterate rows in time order
    for scan_id, tag_id, ts, pred_loc, pred_conf in rows:
        if current_tag is None:
            current_tag = tag_id

        # If tag changes, flush previous segment
        if tag_id != current_tag:
            # close last segment as DWELL if needed
            if stable_loc is not None and stable_start_ts is not None:
                dur = (iso_to_dt(ts) - iso_to_dt(stable_start_ts)).total_seconds()
                if dur >= args.dwell_seconds:
                    avg_conf = stable_conf_sum / max(stable_n, 1)
                    emit_event(current_tag, stable_loc, "DWELL", stable_start_ts, ts, avg_conf)

            # reset state for new tag
            current_tag = tag_id
            buffer_loc = None
            buffer_count = 0
            stable_loc = None
            stable_start_ts = None
            stable_conf_sum = 0.0
            stable_n = 0

        # --- Stability confirmation window ---
        if buffer_loc is None or pred_loc != buffer_loc:
            buffer_loc = pred_loc
            buffer_count = 1
        else:
            buffer_count += 1

        # Not stable enough yet → continue
        if buffer_count < args.min_stable:
            continue

        # Confirmed stable location change
        if stable_loc is None:
            # first stable location
            stable_loc = buffer_loc
            stable_start_ts = ts
            stable_conf_sum = pred_conf
            stable_n = 1
            emit_event(tag_id, stable_loc, "ARRIVAL", ts, None, pred_conf)
            continue

        # If stable location continues, accumulate
        if buffer_loc == stable_loc:
            stable_conf_sum += pred_conf
            stable_n += 1
            continue

        # Otherwise: stable location changed -> DEPARTURE + ARRIVAL, and maybe DWELL
        prev_loc = stable_loc
        prev_start = stable_start_ts
        prev_end = ts

        # DWELL check for the previous stable segment
        dur = (iso_to_dt(prev_end) - iso_to_dt(prev_start)).total_seconds()
        prev_avg_conf = stable_conf_sum / max(stable_n, 1)

        if dur >= args.dwell_seconds:
            emit_event(tag_id, prev_loc, "DWELL", prev_start, prev_end, prev_avg_conf)

        emit_event(tag_id, prev_loc, "DEPARTURE", prev_end, None, prev_avg_conf)

        # Start new stable segment
        stable_loc = buffer_loc
        stable_start_ts = ts
        stable_conf_sum = pred_conf
        stable_n = 1
        emit_event(tag_id, stable_loc, "ARRIVAL", ts, None, pred_conf)

    # Insert events
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
