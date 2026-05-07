from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RoutingDecision(BaseModel):
    expert: str
    strength: float
    refine: bool
    confidence: float
    probabilities: Dict[str, float]
    reason: str
    distortion_summary: Optional[Dict[str, float]] = None
    # Dynamic-routing telemetry. Optional because legacy clients (or runs with
    # system.dynamic_routing=false) may not populate them.
    policy_advice: Optional[Dict[str, Any]] = None
    dynamic_candidates: Optional[List[Dict[str, Any]]] = None
    decision_reason: Optional[str] = None
    dynamic_routing: Optional[bool] = None


class MetricsResult(BaseModel):
    original: Dict[str, Any]
    enhanced: Dict[str, Any]
    improvement: Dict[str, Any]
    similarity_vs_noisy_input: Dict[str, Any]
    vs_clean_reference: Optional[Dict[str, Any]] = None


class EnhancementResponse(BaseModel):
    id: str
    input_path: str
    output_audio_path: str
    metrics_path: str
    routing_log_path: str
    csv_report_path: str
    plots: List[str]
    routing: RoutingDecision
    metrics: MetricsResult
