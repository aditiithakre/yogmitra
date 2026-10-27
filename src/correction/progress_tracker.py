"""
src/correction/progress_tracker.py
───────────────────────────────────
Session history and progress tracking module using SQLite.

Tracks:
- Session start/end timestamp and duration
- Poses practiced during the session
- Frame counts and alignment scores (average & peak)
- Recurring joint-level errors and average deviation magnitudes
- Biomechanical calibration status

Provides lightweight, zero-dependency persistence to data/yoga_sessions.db.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/yoga_sessions.db")


@dataclass
class FrameRecord:
    pose_name: str
    alignment_score: float
    deviations: list[dict[str, Any]]
    is_reliable: bool
    timestamp: float = field(default_factory=time.time)


class ProgressTracker:
    """
    Tracks real-time session metrics and persists session summaries to SQLite.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.session_active = False
        self.start_timestamp: float = 0.0
        self.start_iso: str = ""
        self.calibrated: bool = False

        # Session accumulators
        self.poses_seen: set[str] = set()
        self.frame_count: int = 0
        self.scores: list[float] = []
        self.joint_fault_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.joint_deviation_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        self._init_db()

    def _init_db(self) -> None:
        """Create database tables if they do not exist."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        start_time TEXT NOT NULL,
                        end_time TEXT NOT NULL,
                        duration_sec REAL NOT NULL,
                        poses_practiced TEXT NOT NULL,
                        frames_evaluated INTEGER NOT NULL,
                        avg_alignment_score REAL NOT NULL,
                        peak_alignment_score REAL NOT NULL,
                        calibrated INTEGER NOT NULL DEFAULT 0
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        pose_name TEXT NOT NULL,
                        joint_name TEXT NOT NULL,
                        error_count INTEGER NOT NULL,
                        avg_deviation REAL NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                conn.commit()
        except Exception as exc:
            logger.error("[ProgressTracker] DB init error: %s", exc)

    def start_session(self, calibrated: bool = False) -> None:
        """Start tracking a new yoga session."""
        self.session_active = True
        self.start_timestamp = time.time()
        self.start_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        self.calibrated = calibrated
        self.poses_seen.clear()
        self.frame_count = 0
        self.scores.clear()
        self.joint_fault_counts.clear()
        self.joint_deviation_sums.clear()
        logger.info("[ProgressTracker] Session started at %s (calibrated=%s)", self.start_iso, calibrated)

    def log_frame(
        self,
        pose_name: str,
        alignment_score: float,
        deviations: list[Any],
        is_reliable: bool = True,
    ) -> None:
        """
        Record a single evaluated frame during practice.
        Accepts PostureDeviation objects or dicts.
        """
        if not self.session_active or not pose_name or pose_name == "—":
            return

        self.poses_seen.add(pose_name)
        self.frame_count += 1

        if is_reliable and alignment_score > 0.0:
            self.scores.append(alignment_score)

        for dev in deviations:
            if isinstance(dev, dict):
                joint = dev.get("joint_name") or dev.get("display_name") or "unknown"
                status = dev.get("status", "needs_work")
                mag = float(dev.get("deviation_magnitude", dev.get("deviation", 0.0)))
            else:
                joint = getattr(dev, "joint_name", None) or getattr(dev, "display_name", "unknown")
                status = getattr(dev, "status", "needs_work")
                mag = float(getattr(dev, "deviation_magnitude", getattr(dev, "deviation", 0.0)))

            # Only record as a fault if status is not 'ok' and there is deviation
            if status != "ok" and mag > 0.0:
                self.joint_fault_counts[pose_name][joint] += 1
                self.joint_deviation_sums[pose_name][joint] += abs(mag)

    def end_session(self) -> Optional[dict[str, Any]]:
        """
        End the active session, persist summary to SQLite, and return summary dict.
        """
        if not self.session_active:
            return None

        self.session_active = False
        duration = round(max(0.0, time.time() - self.start_timestamp), 1)
        end_iso = time.strftime("%Y-%m-%d %H:%M:%S")

        avg_score = round(float(sum(self.scores) / len(self.scores)), 1) if self.scores else 0.0
        peak_score = round(float(max(self.scores)), 1) if self.scores else 0.0
        poses_str = ", ".join(sorted(self.poses_seen)) if self.poses_seen else "None"

        summary = {
            "start_time": self.start_iso,
            "end_time": end_iso,
            "duration_sec": duration,
            "poses_practiced": sorted(self.poses_seen),
            "frames_evaluated": self.frame_count,
            "avg_alignment_score": avg_score,
            "peak_alignment_score": peak_score,
            "calibrated": self.calibrated,
            "top_joint_faults": [],
        }

        # If session had negligible activity, skip DB write
        if self.frame_count == 0 and duration < 2.0:
            return summary

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO sessions (
                        start_time, end_time, duration_sec, poses_practiced,
                        frames_evaluated, avg_alignment_score, peak_alignment_score, calibrated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.start_iso,
                        end_iso,
                        duration,
                        poses_str,
                        self.frame_count,
                        avg_score,
                        peak_score,
                        1 if self.calibrated else 0,
                    ),
                )
                session_id = cursor.lastrowid

                # Record joint errors
                fault_records = []
                for p_name, joints in self.joint_fault_counts.items():
                    for j_name, cnt in joints.items():
                        total_deg = self.joint_deviation_sums[p_name][j_name]
                        avg_deg = round(total_deg / max(1, cnt), 1)
                        cursor.execute(
                            """
                            INSERT INTO session_errors (
                                session_id, pose_name, joint_name, error_count, avg_deviation
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (session_id, p_name, j_name, cnt, avg_deg),
                        )
                        fault_records.append({
                            "pose": p_name,
                            "joint": j_name,
                            "count": cnt,
                            "avg_dev_deg": avg_deg,
                        })

                conn.commit()
                # Sort top faults by count descending
                fault_records.sort(key=lambda x: x["count"], reverse=True)
                summary["top_joint_faults"] = fault_records[:5]
                logger.info("[ProgressTracker] Saved session #%s (duration %.1fs, avg %.1f%%)",
                            session_id, duration, avg_score)
        except Exception as exc:
            logger.error("[ProgressTracker] Failed to persist session: %s", exc)

        return summary

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve recent session records from database."""
        if not self.db_path.exists():
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, start_time, duration_sec, poses_practiced,
                           frames_evaluated, avg_alignment_score, peak_alignment_score, calibrated
                    FROM sessions
                    ORDER BY session_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()

                results = []
                for r in rows:
                    s_id = r["session_id"]
                    cursor.execute(
                        """
                        SELECT pose_name, joint_name, error_count, avg_deviation
                        FROM session_errors
                        WHERE session_id = ?
                        ORDER BY error_count DESC
                        LIMIT 3
                        """,
                        (s_id,),
                    )
                    err_rows = cursor.fetchall()
                    top_errors = [
                        f"{er['joint_name']} ({er['error_count']}x, ~{er['avg_deviation']}°)"
                        for er in err_rows
                    ]
                    results.append({
                        "session_id": s_id,
                        "date": r["start_time"],
                        "duration_sec": r["duration_sec"],
                        "poses": r["poses_practiced"],
                        "frames": r["frames_evaluated"],
                        "avg_score": r["avg_alignment_score"],
                        "peak_score": r["peak_alignment_score"],
                        "calibrated": bool(r["calibrated"]),
                        "top_faults": top_errors,
                    })
                return results
        except Exception as exc:
            logger.error("[ProgressTracker] History fetch error: %s", exc)
            return []

    def format_history_table(self, limit: int = 10) -> str:
        """Return a formatted text table of past sessions."""
        records = self.get_history(limit=limit)
        if not records:
            return "No practice sessions recorded yet. Start practicing with Yoga Mitra to build history!"

        lines = [
            "=" * 78,
            f"{'ID':<4} | {'Date & Time':<19} | {'Dur(s)':<6} | {'Avg%':<5} | {'Peak%':<5} | {'Poses':<24}",
            "-" * 78,
        ]
        for r in records:
            poses = r["poses"]
            if len(poses) > 24:
                poses = poses[:21] + "..."
            lines.append(
                f"{r['session_id']:<4} | {r['date']:<19} | {r['duration_sec']:<6.0f} | "
                f"{r['avg_score']:<5.1f} | {r['peak_score']:<5.1f} | {poses:<24}"
            )
            if r["top_faults"]:
                fault_str = "       Top recurring issues: " + "; ".join(r["top_faults"])
                lines.append(fault_str)
        lines.append("=" * 78)
        return "\n".join(lines)

    def get_longitudinal_summary(self, limit: int = 15) -> dict[str, Any]:
        """
        Computes longitudinal progress analytics across multiple practice sessions:
        - Overall score improvement trends (early sessions vs recent sessions)
        - Repeated joint-level error tracking and deviation reduction
        - Calibration usage rate
        - Grounded personalized recommendations
        """
        records = self.get_history(limit=limit)
        if not records:
            return {
                "total_sessions": 0,
                "status": "no_data",
                "message": "No sessions recorded yet.",
            }

        # Order chronologically (oldest to newest)
        chrono = sorted(records, key=lambda x: x["session_id"])
        scores = [r["avg_score"] for r in chrono if r["avg_score"] > 0]
        total_sec = sum(r["duration_sec"] for r in chrono)
        total_frames = sum(r["frames"] for r in chrono)
        calibrated_count = sum(1 for r in chrono if r["calibrated"])

        # Improvement delta: compare first half vs second half if >= 2 sessions
        improvement_delta = 0.0
        if len(scores) >= 2:
            mid = len(scores) // 2
            older_avg = float(np.mean(scores[:mid]))
            recent_avg = float(np.mean(scores[mid:]))
            improvement_delta = round(recent_avg - older_avg, 1)

        # Aggregate joint errors across sessions to detect recurring mistakes
        joint_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "deviations": [], "poses": set()})
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT e.pose_name, e.joint_name, e.error_count, e.avg_deviation
                    FROM session_errors e
                    JOIN sessions s ON e.session_id = s.session_id
                    ORDER BY s.session_id DESC
                    LIMIT 100
                    """
                )
                for p_name, j_name, cnt, avg_dev in cursor.fetchall():
                    joint_stats[j_name]["count"] += int(cnt)
                    joint_stats[j_name]["deviations"].append(float(avg_dev))
                    joint_stats[j_name]["poses"].add(p_name)
        except Exception as exc:
            logger.error("[ProgressTracker] Joint stats aggregation error: %s", exc)

        recurring_issues = []
        for j_name, data in sorted(joint_stats.items(), key=lambda x: -x[1]["count"]):
            if data["count"] > 0:
                mean_dev = round(float(np.mean(data["deviations"])), 1) if data["deviations"] else 0.0
                recurring_issues.append({
                    "joint": j_name,
                    "total_occurrences": data["count"],
                    "mean_deviation_deg": mean_dev,
                    "poses": sorted(list(data["poses"])),
                })

        # Grounded personalized recommendations based on recurring errors
        recommendations = []
        if recurring_issues:
            top = recurring_issues[0]
            j_clean = top["joint"].lower()
            if "knee" in j_clean:
                recommendations.append(
                    f"Frequent {top['joint']} deviation (~{top['mean_deviation_deg']}° across {len(top['poses'])} poses). "
                    "Focus on tracking your knee directly over your second toe without collapsing inward."
                )
            elif "arm" in j_clean or "elbow" in j_clean:
                recommendations.append(
                    f"Recurring {top['joint']} misalignment observed. Maintain active extension through fingertips "
                    "while avoiding locking elbows."
                )
            elif "shoulder" in j_clean:
                recommendations.append(
                    f"Repeated {top['joint']} elevation detected. Consciously roll shoulders down and back, away from ears."
                )
            else:
                recommendations.append(
                    f"Primary recurring focus area is {top['joint']} (avg deviation {top['mean_deviation_deg']}°)."
                )

        if calibrated_count == 0:
            recommendations.append("Run Mountain Pose calibration (press C in app) to adapt posture ranges to your limb proportions.")
        elif improvement_delta > 0:
            recommendations.append(f"Positive progress! Your average posture alignment has improved by +{improvement_delta}% over recent sessions.")

        return {
            "total_sessions": len(records),
            "total_duration_sec": round(total_sec, 1),
            "total_frames_evaluated": total_frames,
            "overall_avg_score": round(float(np.mean(scores)), 1) if scores else 0.0,
            "improvement_delta": improvement_delta,
            "calibrated_sessions_rate": round(calibrated_count / max(1, len(records)), 2),
            "recurring_joint_issues": recurring_issues[:4],
            "personalized_recommendations": recommendations,
        }

    def format_progress_report(self, limit: int = 15) -> str:
        """Return a formatted longitudinal progress report."""
        summary = self.get_longitudinal_summary(limit=limit)
        if summary.get("status") == "no_data":
            return summary.get("message", "No practice data available.")

        delta_sign = "+" if summary["improvement_delta"] > 0 else ""
        lines = [
            "=" * 78,
            "  YOGA MITRA — LONGITUDINAL PROGRESS & PERSONALIZATION REPORT",
            "=" * 78,
            f"  Total Sessions Tracked:      {summary['total_sessions']}",
            f"  Total Practice Time:         {summary['total_duration_sec'] / 60:.1f} minutes ({summary['total_duration_sec']:.0f}s)",
            f"  Total Evaluated Frames:      {summary['total_frames_evaluated']}",
            f"  Overall Average Score:       {summary['overall_avg_score']:.1f}%",
            f"  Recent Improvement Trend:    {delta_sign}{summary['improvement_delta']:.1f}%",
            f"  Calibrated Practice Rate:    {summary['calibrated_sessions_rate']*100:.0f}%",
            "-" * 78,
            "  RECURRING JOINT MISALIGNMENTS:",
        ]

        if summary["recurring_joint_issues"]:
            for issue in summary["recurring_joint_issues"]:
                poses_str = ", ".join(issue["poses"])
                lines.append(
                    f"    • {issue['joint']:<18} | Occurrences: {issue['total_occurrences']:>3}x | "
                    f"Avg Dev: ~{issue['mean_deviation_deg']:>4.1f}° | Poses: {poses_str}"
                )
        else:
            lines.append("    • No recurring joint deviations detected! Excellent form stability.")

        lines.append("-" * 78)
        lines.append("  PERSONALIZED COACHING RECOMMENDATIONS:")
        for rec in summary["personalized_recommendations"]:
            lines.append(f"    ▶ {rec}")

        lines.append("=" * 78)
        return "\n".join(lines)

