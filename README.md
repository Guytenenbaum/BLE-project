# BLE RSSI 

**Goal:** Ingest BLE RSSI scans into SQL, train an ML model to predict zone/location from RSSI, write predictions back to SQL, and generate operational events (ARRIVAL/DEPARTURE/DWELL/MISSING_READS).

## Architecture
Raw scans (RSSI) -> SQL warehouse -> ML zone classifier -> smoothed predictions -> event generation -> analytics queries.

## Repo structure
- sql/        Schema + analytics queries
- etl/        Load/clean dataset into SQL tables
- ml/         Train/evaluate model + write predictions to SQL
- events/     Create ARRIVAL/DEPARTURE/DWELL/MISSING_READS from predictions
- figures/    Plots (confusion matrix, diagrams)
- data/       Local dataset files (gitignored)
