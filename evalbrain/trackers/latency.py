from typing import List, Dict
from evalbrain.models import Trace


class LatencyTracker:
    """Utility class to aggregate and calculate latency metrics."""
    
    @staticmethod
    def calculate_percentiles(latencies: List[float], percentiles: List[float] = None) -> Dict[str, float]:
        """
        Calculate specified percentiles for a list of latencies.
        Default percentiles: p50, p90, p95, p99.
        """
        if percentiles is None:
            percentiles = [50, 90, 95, 99]
            
        if not latencies:
            return {f"p{p}": 0.0 for p in percentiles}
            
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        results = {}
        
        for p in percentiles:
            idx = int((p / 100.0) * n)
            # handle boundary
            if idx >= n:
                idx = n - 1
            results[f"p{p}"] = sorted_latencies[idx]
            
        return results

    @staticmethod
    def get_span_latencies(traces: List[Trace], span_name: str = None) -> List[float]:
        """
        Extract all latencies (in ms) from a list of traces, optionally filtered by span name.
        """
        latencies = []
        for trace in traces:
            for span in trace.spans:
                if span.latency_ms is not None:
                    if span_name is None or span.name == span_name:
                        latencies.append(span.latency_ms)
        return latencies

    @staticmethod
    def aggregate_trace_latencies(traces: List[Trace], span_name: str = None) -> Dict[str, float]:
        """
        Calculate latency percentiles across all spans in the provided traces.
        """
        latencies = LatencyTracker.get_span_latencies(traces, span_name)
        return LatencyTracker.calculate_percentiles(latencies)
