import os
import importlib
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from evalbrain.core.tracer import EvalBrain
from evalbrain.models import Trace, PromptVersion

app = FastAPI(
    title="EvalBrain API",
    description="REST API for EvalBrain observability and regression testing.",
    version="0.1.0",
)

def get_brain() -> EvalBrain:
    # Instantiate the global brain
    return EvalBrain()

class RegressionRunRequest(BaseModel):
    suite: str
    target: str
    evaluators: Optional[List[str]] = None

@app.get("/traces", response_model=List[Trace])
def list_traces(
    project: Optional[str] = None,
    limit: int = Query(50, ge=1, le=1000)
):
    """List recent traces, optionally filtered by project."""
    brain = get_brain()
    traces = brain.get_traces()
    if project:
        traces = [t for t in traces if t.project == project]
    return traces[-limit:]

@app.get("/traces/{trace_id}", response_model=Trace)
def get_trace(trace_id: str):
    """Get a single trace detail."""
    brain = get_brain()
    traces = brain.get_traces()
    for t in traces:
        if t.trace_id.startswith(trace_id):
            return t
    raise HTTPException(status_code=404, detail="Trace not found")

@app.get("/metrics/summary")
def get_metrics_summary(project: Optional[str] = None):
    """Aggregated metrics for a project or all traces."""
    brain = get_brain()
    traces = brain.get_traces()
    if project:
        traces = [t for t in traces if t.project == project]
        
    total_traces = len(traces)
    total_spans = sum(len(t.spans) for t in traces)
    
    # Calculate average faithfulness or other metrics if available
    # For now, providing basic aggregations.
    return {
        "total_traces": total_traces,
        "total_spans": total_spans,
    }

@app.get("/metrics/cost")
def get_metrics_cost(project: Optional[str] = None):
    """Cost metrics over time or by model."""
    brain = get_brain()
    traces = brain.get_traces()
    if project:
        traces = [t for t in traces if t.project == project]
        
    total_cost = 0.0
    for t in traces:
        for span in t.spans:
            if span.cost_usd:
                total_cost += span.cost_usd
                
    return {
        "total_cost_usd": total_cost,
    }

@app.post("/regression/run")
def run_regression_suite(req: RegressionRunRequest):
    """Trigger a regression suite run."""
    try:
        module_path, func_name = req.target.split(":")
        module = importlib.import_module(module_path)
        target_func = getattr(module, func_name)
    except ValueError:
        raise HTTPException(status_code=400, detail="Target must be in format 'module:function'")
    except ImportError as e:
        raise HTTPException(status_code=400, detail=f"Error loading module: {e}")
    except AttributeError:
        raise HTTPException(status_code=400, detail=f"Function '{func_name}' not found in '{module_path}'")

    brain = get_brain()
    try:
        # Run suite synchronously for now. In a real system, you might want this to be async/background.
        brain.run_suite(req.suite, target_func, req.evaluators)
        return {"status": "success", "message": f"Regression suite {req.suite} completed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running suite: {e}")

@app.get("/prompts")
def list_prompts():
    """List prompt versions."""
    # This requires registry backend hookup
    return {"message": "Prompt listing from registry is pending storage backend hookup."}

@app.get("/prompts/{prompt_id}/diff")
def get_prompt_diff(prompt_id: str, v1: str, v2: str):
    """Get prompt diff between two versions."""
    return {"message": f"Diffing {v1} and {v2} for {prompt_id} is pending registry backend hookup."}

# Mount static files for dashboard (Step 14)
dashboard_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
if os.path.isdir(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

@app.get("/")
def read_root():
    """Root redirect to docs or dashboard."""
    return {"message": "Welcome to EvalBrain API. Visit /docs for API documentation or /dashboard for the web UI."}

