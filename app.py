import os
import json
import math
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from dateutil import parser

load_dotenv()

app = Flask(__name__)
PORT = int(os.getenv('PORT', 5000))

# Default weights for scoring
DEFAULT_WEIGHTS = {
    "pm_health": 0.30,
    "jobcard": 0.15,
    "fitness": 0.20,
    "branding": 0.10,
    "mileage": 0.10,
    "cleaning": 0.05,
    "stabling": 0.10
}

# ==================== UTILITY FUNCTIONS ====================

def minutes_until(date_str, now):
    """Calculate minutes until a given datetime string"""
    if not date_str:
        return -999
    dt = parser.parse(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int((dt - now).total_seconds() / 60)

def convert_snapshot_to_dataframe(snapshot_data):
    """Convert JSON snapshot to induction input dataframe"""
    now = datetime.now(timezone.utc)
    trains = {}
    
    # Process branding priorities
    for b in snapshot_data.get("branding_priorities", []):
        tid = b["train_id"]
        trains.setdefault(tid, {})
        if b.get("priority_level") == 1:
            score = 1.0
        elif b.get("priority_level") == 2:
            score = 0.6
        else:
            score = 0.3
        trains[tid]["branding_score"] = max(trains[tid].get("branding_score", 0), score)
    
    # Process cleaning slots
    for c in snapshot_data.get("cleaning_slots", []):
        tid = c["train_id"]
        trains.setdefault(tid, {})
        start = parser.parse(c["slot_start"])
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        within_24h = (start - now).total_seconds() <= 24*3600 and (start - now).total_seconds() >= -3600
        if within_24h:
            trains[tid]["cleaning_required"] = 1
        else:
            trains[tid].setdefault("cleaning_required", trains[tid].get("cleaning_required", 0))
    
    # Process stabling geometry
    for s in snapshot_data.get("stabling_geometry", []):
        tid = s["train_id"]
        trains.setdefault(tid, {})
        dist = float(s.get("distance_from_buffer_m", 0))
        track = float(s.get("track_no", 0))
        penalty = int(dist*3 + track)
        trains[tid]["stabling_penalty"] = penalty
    
    # Process fitness certificates
    for f in snapshot_data.get("fitness_certificates", []):
        tid = f["train_id"]
        trains.setdefault(tid, {})
        vals = []
        for k in ["rolling_stock_validity", "signalling_validity", "telecom_validity"]:
            v = f.get(k)
            if v:
                vals.append(minutes_until(v, now))
        if vals:
            trains[tid]["minutes_to_latest_fitness_expiry"] = min(vals)
    
    # Process job cards
    job_counts = {}
    for j in snapshot_data.get("job_card_status", []):
        tid = j["train_id"]
        job_counts.setdefault(tid, {"total": 0, "open": 0})
        job_counts[tid]["total"] += 1
        if j.get("status", "").lower() in ("pending", "open"):
            job_counts[tid]["open"] += 1
    
    for tid, counts in job_counts.items():
        trains.setdefault(tid, {})
        trains[tid]["jobcard_open_frac"] = counts["open"] / max(1, counts["total"])
    
    # Process mileage
    mileage_map = {}
    for m in snapshot_data.get("mileage", []):
        tid = m["train_id"]
        mileage_map[tid] = m.get("delta_km", 0)
    
    # Collect all train IDs
    all_train_ids = set(trains.keys()) | set(mileage_map.keys()) | set(job_counts.keys())
    for b in snapshot_data.get("branding_priorities", []):
        all_train_ids.add(b["train_id"])
    
    # Build rows
    rows = []
    for tid in sorted(all_train_ids):
        rec = trains.get(tid, {})
        delta_km = mileage_map.get(tid, 0)
        rows.append({
            "trainset": tid,
            "pm_failure_prob": 0.10,  # Default value
            "jobcard_open_frac": rec.get("jobcard_open_frac", 0.0),
            "minutes_to_latest_fitness_expiry": rec.get("minutes_to_latest_fitness_expiry", -999),
            "branding_score": rec.get("branding_score", 0.0),
            "mileage_need_raw": delta_km,
            "cleaning_required": rec.get("cleaning_required", 0),
            "stabling_penalty": rec.get("stabling_penalty", 10),
            "manual_force_in": 0,
            "manual_force_out": 0
        })
    
    # Normalize mileage
    vals = [r["mileage_need_raw"] for r in rows]
    if max(vals) == min(vals):
        for r in rows:
            r["mileage_need"] = 0.0
    else:
        mn, mx = min(vals), max(vals)
        for r in rows:
            r["mileage_need"] = (r["mileage_need_raw"] - mn) / (mx - mn) if mx > mn else 0.0
    
    # Remove raw mileage
    for r in rows:
        del r["mileage_need_raw"]
    
    return pd.DataFrame(rows)

def normalize_series(s):
    """Normalize a pandas series to 0-1"""
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def compute_scores(df, weights):
    """Compute composite scores for each train"""
    df = df.copy()
    
    # Transform scores (higher is better)
    df["score_pm_health"] = 1.0 - df["pm_failure_prob"].clip(0, 1)
    df["score_jobcard"] = 1.0 - df["jobcard_open_frac"].clip(0, 1)
    df["score_fitness_raw"] = df["minutes_to_latest_fitness_expiry"].clip(lower=-120)
    df["score_branding"] = df["branding_score"].clip(0, 1)
    df["score_mileage"] = df["mileage_need"].clip(0, 1)
    df["score_cleaning"] = 1.0 - df["cleaning_required"].clip(0, 1)
    
    # Normalize fitness and stabling
    df["score_fitness"] = normalize_series(df["score_fitness_raw"])
    df["score_stabling"] = 1.0 - normalize_series(df["stabling_penalty"].astype(float))
    
    # Composite score
    df["composite_score"] = (
        weights["pm_health"] * df["score_pm_health"] +
        weights["jobcard"] * df["score_jobcard"] +
        weights["fitness"] * df["score_fitness"] +
        weights["branding"] * df["score_branding"] +
        weights["mileage"] * df["score_mileage"] +
        weights["cleaning"] * df["score_cleaning"] +
        weights["stabling"] * df["score_stabling"]
    )
    
    # Contributions for explainability
    df["contrib_pm"] = (weights["pm_health"] * df["score_pm_health"]).round(4)
    df["contrib_jobcard"] = (weights["jobcard"] * df["score_jobcard"]).round(4)
    df["contrib_fitness"] = (weights["fitness"] * df["score_fitness"]).round(4)
    df["contrib_branding"] = (weights["branding"] * df["score_branding"]).round(4)
    df["contrib_mileage"] = (weights["mileage"] * df["score_mileage"]).round(4)
    df["contrib_cleaning"] = (weights["cleaning"] * df["score_cleaning"]).round(4)
    df["contrib_stabling"] = (weights["stabling"] * df["score_stabling"]).round(4)
    df["composite_score"] = df["composite_score"].round(6)
    
    return df

def detect_conflicts(df, pm_hard_threshold):
    """Detect conflicts in the train data"""
    alerts = []
    for _, r in df.iterrows():
        if r["minutes_to_latest_fitness_expiry"] < 0 and not bool(r.get("manual_force_in", 0)):
            alerts.append({"trainset": r["trainset"], "alert": "FITNESS_EXPIRED"})
        if r["pm_failure_prob"] > pm_hard_threshold and not bool(r.get("manual_force_in", 0)):
            alerts.append({"trainset": r["trainset"], "alert": "HIGH_PM_RISK"})
        if r["jobcard_open_frac"] > 0.7:
            alerts.append({"trainset": r["trainset"], "alert": "HIGH_JOBCARD_OPEN"})
    return alerts

def greedy_select(sorted_df, num_to_induct, cleaning_capacity, pm_hard_threshold, allow_expired):
    """Select trains for induction using greedy algorithm"""
    selected = []
    cleaning_used = 0
    
    for _, r in sorted_df.iterrows():
        if len(selected) >= num_to_induct:
            break
        
        # Manual force-outs skip
        if bool(r.get("manual_force_out", False)):
            continue
        
        # Forced in always include
        if bool(r.get("manual_force_in", False)):
            selected.append({
                "trainset": str(r["trainset"]),
                "reason": "FORCED_IN",
                "composite_score": float(r["composite_score"])
            })
            continue
        
        # Hard constraints
        if (r["minutes_to_latest_fitness_expiry"] < 0) and (not allow_expired):
            continue
        if (r["pm_failure_prob"] > pm_hard_threshold):
            continue
        
        # Cleaning capacity
        if int(r["cleaning_required"]) == 1 and cleaning_used >= cleaning_capacity:
            continue
        if int(r["cleaning_required"]) == 1:
            cleaning_used += 1
        
        selected.append({
            "trainset": str(r["trainset"]),
            "reason": "SELECTED",
            "composite_score": float(r["composite_score"])
        })
    
    return selected, cleaning_used

# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Train Induction Simulator"}), 200

@app.route('/simulate', methods=['POST'])
def simulate_induction():
    """
    Main simulation endpoint
    
    Accepts JSON snapshot data and simulation parameters
    Returns ranked trains and selected trains for induction
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract parameters
        snapshot = data.get("snapshot", {})
        num_to_induct = data.get("num_to_induct", 6)
        cleaning_capacity = data.get("cleaning_capacity", 2)
        pm_threshold = data.get("pm_threshold", 0.8)
        allow_expired = data.get("allow_expired", False)
        weights = data.get("weights", DEFAULT_WEIGHTS)
        
        # Validate snapshot
        if not snapshot:
            return jsonify({"error": "Snapshot data is required"}), 400
        
        # Convert snapshot to dataframe
        df = convert_snapshot_to_dataframe(snapshot)
        
        if df.empty:
            return jsonify({"error": "No trains found in snapshot"}), 400
        
        # Compute scores
        df_scored = compute_scores(df, weights)
        
        # Sort by composite score
        df_sorted = df_scored.sort_values("composite_score", ascending=False).reset_index(drop=True)
        
        # Select trains
        selected, cleaning_used = greedy_select(
            df_sorted, num_to_induct, cleaning_capacity, pm_threshold, allow_expired
        )
        
        # Calculate KPIs
        expected_withdrawals = float(sum([
            df.loc[df["trainset"] == s["trainset"], "pm_failure_prob"].iloc[0]
            for s in selected
        ])) if selected else 0.0
        
        total_stabling = float(sum([
            df.loc[df["trainset"] == s["trainset"], "stabling_penalty"].iloc[0]
            for s in selected
        ])) if selected else 0.0
        
        branding_sum = float(sum([
            df.loc[df["trainset"] == s["trainset"], "branding_score"].iloc[0]
            for s in selected
        ])) if selected else 0.0
        
        # Detect conflicts
        conflicts = detect_conflicts(df, pm_threshold)
        
        # Prepare ranked output
        cols_to_return = [
            "trainset", "composite_score", "pm_failure_prob", "jobcard_open_frac",
            "minutes_to_latest_fitness_expiry", "branding_score", "mileage_need",
            "cleaning_required", "stabling_penalty",
            "contrib_pm", "contrib_jobcard", "contrib_fitness", "contrib_branding",
            "contrib_mileage", "contrib_cleaning", "contrib_stabling"
        ]
        
        available_cols = [c for c in cols_to_return if c in df_sorted.columns]
        ranked = df_sorted[available_cols].to_dict(orient='records')
        
        # Build response
        response = {
            "success": True,
            "summary": {
                "num_to_induct_requested": num_to_induct,
                "selected_count": len(selected),
                "cleaning_capacity": cleaning_capacity,
                "cleaning_used": cleaning_used,
                "pm_threshold": pm_threshold,
                "allow_expired": allow_expired,
                "expected_withdrawals": round(expected_withdrawals, 3),
                "total_stabling_cost": round(total_stabling, 1),
                "branding_exposure": round(branding_sum, 3),
                "conflicts_count": len(conflicts)
            },
            "selected_trains": selected,
            "ranked_trains": ranked[:20],  # Return top 20
            "conflicts": conflicts[:10],  # Return first 10 conflicts
            "weights_used": weights
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/convert', methods=['POST'])
def convert_snapshot():
    """
    Convert JSON snapshot to CSV format
    
    Accepts JSON snapshot data
    Returns converted data in CSV-ready format
    """
    try:
        data = request.get_json()
        
        if not data or "snapshot" not in data:
            return jsonify({"error": "Snapshot data is required"}), 400
        
        # Convert snapshot to dataframe
        df = convert_snapshot_to_dataframe(data["snapshot"])
        
        # Convert to list of records
        records = df.to_dict(orient='records')
        
        return jsonify({
            "success": True,
            "train_count": len(records),
            "data": records
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        "service": "Train Induction Simulator API",
        "version": "1.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/simulate": "POST - Run induction simulation with snapshot data",
            "/convert": "POST - Convert snapshot JSON to CSV format",
            "/": "GET - API information"
        },
        "simulate_parameters": {
            "snapshot": "JSON object with train data (required)",
            "num_to_induct": "Number of trains to select (default: 6)",
            "cleaning_capacity": "Cleaning slots available (default: 2)",
            "pm_threshold": "PM failure probability threshold (default: 0.8)",
            "allow_expired": "Allow expired fitness certificates (default: false)",
            "weights": "Custom scoring weights (optional)"
        }
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)