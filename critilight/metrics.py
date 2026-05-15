import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


def parse_tripinfo_metrics(file_path):
    if not os.path.exists(file_path):
        return {"durations": [], "waiting_times": []}, {"durations": [], "waiting_times": []}, 0, 0

    tree = ET.parse(file_path)
    root = tree.getroot()

    completed = {"durations": [], "waiting_times": []}
    all_metrics = {"durations": [], "waiting_times": []}
    completed_count = 0
    total_count = 0

    for trip in root.findall("tripinfo"):
        total_count += 1
        duration = float(trip.get("duration", 0))
        waiting_time = float(trip.get("waitingTime", 0))
        all_metrics["durations"].append(duration)
        all_metrics["waiting_times"].append(waiting_time)

        if float(trip.get("arrival", -1)) != -1:
            completed_count += 1
            completed["durations"].append(duration)
            completed["waiting_times"].append(waiting_time)

    return completed, all_metrics, completed_count, total_count


def parse_queue_export(file_path):
    if not os.path.exists(file_path):
        return 0.0

    tree = ET.parse(file_path)
    root = tree.getroot()
    queue_lengths = [
        float(lane.get("queueing_length", 0))
        for data in root.findall("data")
        for lanes in data.findall("lanes")
        for lane in lanes.findall("lane")
    ]
    return sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0.0


def Cal_Offline(
    avg_queue,
    scenario_name,
    method_name,
    experiment_name,
    note,
    tripinfo_file,
    queue_export_file,
    log_file=None,
):
    completed, all_metrics, completed_count, total_count = parse_tripinfo_metrics(tripinfo_file)
    avg_queue_length = parse_queue_export(queue_export_file)

    att_complete = sum(completed["durations"]) / len(completed["durations"]) if completed["durations"] else 0.0
    awt_complete = sum(completed["waiting_times"]) / len(completed["waiting_times"]) if completed["waiting_times"] else 0.0
    att_all = sum(all_metrics["durations"]) / len(all_metrics["durations"]) if all_metrics["durations"] else 0.0
    awt_all = sum(all_metrics["waiting_times"]) / len(all_metrics["waiting_times"]) if all_metrics["waiting_times"] else 0.0
    completed_percent = (completed_count / total_count * 100.0) if total_count else 0.0

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 80,
        f"Offline report | time={timestamp} | scenario={scenario_name} | method={method_name} | experiment={experiment_name} | note={note}",
        "=" * 80,
        f"ATT (completed): {att_complete:.2f} s",
        f"AWT (completed): {awt_complete:.2f} s",
        f"ATT (all): {att_all:.2f} s",
        f"AWT (all): {awt_all:.2f} s",
        f"AQL: {avg_queue_length:.2f} m",
        f"AQN: {avg_queue:.2f}",
        f"Completion: {completed_count} / {total_count} ({completed_percent:.2f}%)",
    ]

    log_target = Path(log_file) if log_file else Path(__file__).resolve().parents[1] / "outputs" / "critilight" / "offline_report.log"
    log_target.parent.mkdir(parents=True, exist_ok=True)
    with open(log_target, "a", encoding="utf-8") as handle:
        for line in lines:
            print(line)
            handle.write(line + "\n")
