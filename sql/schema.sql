-- Core entities
CREATE TABLE IF NOT EXISTS scans (
  scan_id      INTEGER PRIMARY KEY,
  tag_id       TEXT NOT NULL,
  ts           TEXT NOT NULL,          -- store ISO timestamp string
  true_loc     TEXT,                   -- e.g., A3 (nullable for unlabeled)
  split        TEXT                    -- train/valid/test (optional)
);

-- Long-format RSSI readings: one row per (scan, beacon)
CREATE TABLE IF NOT EXISTS raw_reads (
  scan_id     INTEGER NOT NULL,
  beacon_id   INTEGER NOT NULL,        -- 1..13
  rssi        REAL NOT NULL,
  PRIMARY KEY (scan_id, beacon_id),
  FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
);

-- Model predictions written back to SQL
CREATE TABLE IF NOT EXISTS loc_predictions (
  scan_id        INTEGER PRIMARY KEY,
  model_version  TEXT NOT NULL,
  pred_loc       TEXT NOT NULL,
  pred_conf      REAL NOT NULL,
  FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
);

-- Event output (what you'd expose to apps/ops)
CREATE TABLE IF NOT EXISTS events (
  event_id     INTEGER PRIMARY KEY,
  tag_id       TEXT NOT NULL,
  loc          TEXT NOT NULL,
  event_type   TEXT NOT NULL,          -- ARRIVAL / DEPARTURE / DWELL / MISSING_READS
  start_ts     TEXT NOT NULL,
  end_ts       TEXT,
  confidence   REAL
);
