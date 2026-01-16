-- 1) Sample: compare true vs predicted
SELECT s.ts, s.true_loc, p.pred_loc, ROUND(p.pred_conf, 3) AS conf
FROM scans s
         JOIN loc_predictions p ON p.scan_id = s.scan_id
WHERE p.model_version = 'logreg_v1'
ORDER BY s.ts
    LIMIT 20;

-- 2) Low-confidence scans (triage)
SELECT s.scan_id, s.ts, p.pred_loc, p.pred_conf
FROM scans s
         JOIN loc_predictions p ON p.scan_id = s.scan_id
WHERE p.model_version = 'logreg_v1' AND p.pred_conf < 0.55
ORDER BY p.pred_conf ASC
    LIMIT 50;

-- 3) Transition matrix (where it moves)
WITH ordered AS (
    SELECT s.tag_id, s.ts, p.pred_loc,
           LAG(p.pred_loc) OVER (PARTITION BY s.tag_id ORDER BY s.ts) AS prev_loc
    FROM scans s
             JOIN loc_predictions p ON p.scan_id = s.scan_id
    WHERE p.model_version = 'logreg_v1'
)
SELECT prev_loc, pred_loc, COUNT(*) AS n
FROM ordered
WHERE prev_loc IS NOT NULL AND prev_loc <> pred_loc
GROUP BY 1,2
ORDER BY n DESC
    LIMIT 30;

-- 4) Dwell events summary
SELECT loc,
       COUNT(*) AS dwell_events,
       AVG((julianday(end_ts) - julianday(start_ts)) * 86400.0) AS avg_dwell_seconds
FROM events
WHERE event_type = 'DWELL' AND end_ts IS NOT NULL
GROUP BY loc
ORDER BY dwell_events DESC;

-- 5) Timeline: events in order
SELECT event_id, tag_id, event_type, loc, start_ts, end_ts, ROUND(confidence, 3) AS conf
FROM events
ORDER BY event_id
    LIMIT 30;
