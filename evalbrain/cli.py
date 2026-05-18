import importlib
import json
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from evalbrain.core.tracer import EvalBrain

app = typer.Typer(help="EvalBrain CLI: Observability and Regression Testing for LLMs.")
prompt_app = typer.Typer(help="Manage prompt versions.")
cost_app = typer.Typer(help="Manage cost tracking.")

app.add_typer(prompt_app, name="prompt")
app.add_typer(cost_app, name="cost")

server_app = typer.Typer(help="Manage API server.")
app.add_typer(server_app, name="server")

console = Console()

def get_brain() -> EvalBrain:
    return EvalBrain()

@app.command("run")
def run_suite(
    suite: str = typer.Argument(..., help="Path to the regression suite file (YAML/JSON)"),
    target: str = typer.Option(..., help="Target function to evaluate, e.g. my_app.main:predict"),
    evaluators: Optional[List[str]] = typer.Option(None, help="List of evaluators to use")
):
    """Run a regression test suite."""
    console.print(f"[bold green]Running regression suite:[/bold green] {suite}")
    
    try:
        module_path, func_name = target.split(":")
        module = importlib.import_module(module_path)
        target_func = getattr(module, func_name)
    except ValueError:
        console.print("[bold red]Error:[/bold red] Target must be in format 'module:function'")
        raise typer.Exit(code=1)
    except ImportError as e:
        console.print(f"[bold red]Error loading module:[/bold red] {e}")
        raise typer.Exit(code=1)
    except AttributeError:
        console.print(f"[bold red]Error:[/bold red] Function '{func_name}' not found in '{module_path}'")
        raise typer.Exit(code=1)

    brain = get_brain()
    
    try:
        brain.run_suite(suite, target_func, evaluators)
        console.print("[bold green]Regression suite completed.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error running suite:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command("traces")
def list_traces(limit: int = typer.Option(10, help="Number of traces to show")):
    """List recent traces."""
    brain = get_brain()
    traces = brain.get_traces()
    
    if not traces:
        console.print("[yellow]No traces found. (Storage backend might not be fully configured yet)[/yellow]")
        return
        
    table = Table("ID", "Project", "Spans", "Created At")
    for t in traces[-limit:]:
        table.add_row(
            str(t.trace_id)[:8],
            t.project,
            str(len(t.spans)),
            str(t.created_at)
        )
    
    console.print(table)

@app.command("trace")
def inspect_trace(trace_id: str = typer.Argument(..., help="ID of the trace to inspect")):
    """Inspect a specific trace."""
    brain = get_brain()
    traces = brain.get_traces()
    
    target_trace = next((t for t in traces if t.trace_id.startswith(trace_id)), None)
    if not target_trace:
        console.print(f"[bold red]Trace {trace_id} not found.[/bold red]")
        raise typer.Exit(code=1)
        
    console.print(f"[bold blue]Trace ID:[/bold blue] {target_trace.trace_id}")
    console.print(f"[bold blue]Project:[/bold blue] {target_trace.project}")
    
    table = Table("Span ID", "Name", "Latency (ms)", "Input Tokens", "Output Tokens", "Cost")
    for span in target_trace.spans:
        table.add_row(
            str(span.span_id)[:8],
            span.name,
            f"{span.latency_ms:.2f}" if span.latency_ms else "-",
            str(span.token_counts.get("input", "-")) if span.token_counts else "-",
            str(span.token_counts.get("output", "-")) if span.token_counts else "-",
            f"${span.cost_usd:.4f}" if span.cost_usd else "-"
        )
    console.print(table)

@prompt_app.command("list")
def prompt_list():
    """List all prompt versions."""
    console.print("[yellow]Prompt listing from registry is pending storage backend hookup.[/yellow]")

@prompt_app.command("diff")
def prompt_diff(v1: str, v2: str):
    """Show prompt diff."""
    console.print(f"[yellow]Diffing {v1} and {v2} is pending registry backend hookup.[/yellow]")

@cost_app.command("summary")
def cost_summary():
    """Show cost breakdown."""
    console.print("[yellow]Cost summary is pending storage backend hookup.[/yellow]")

@app.command("export")
def export_traces(format: str = typer.Argument("json", help="Format to export (json/csv)")):
    """Export traces to JSON/CSV."""
    brain = get_brain()
    traces = brain.get_traces()
    
    if format.lower() == "json":
        data = [t.model_dump(mode="json") for t in traces]
        console.print(json.dumps(data, indent=2))
    elif format.lower() == "csv":
        console.print("[yellow]CSV export not yet implemented.[/yellow]")
    else:
        console.print(f"[bold red]Unknown format:[/bold red] {format}")

def _run_server(host: str, port: int, reload: bool):
    console.print(f"[bold green]Starting EvalBrain API Server on {host}:{port}...[/bold green]")
    try:
        import uvicorn
        uvicorn.run("evalbrain.server.app:app", host=host, port=port, reload=reload)
    except ImportError:
        console.print("[bold red]Uvicorn is not installed. Please install evalbrain[server][/bold red]")
        raise typer.Exit(code=1)

@server_app.command("start")
def start_server(
    host: str = typer.Option("127.0.0.1", help="Host IP to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(True, help="Enable auto-reload")
):
    """Launch the FastAPI API server and Dashboard."""
    _run_server(host, port, reload)

@app.command("dashboard")
def start_dashboard():
    """Launch the web UI dashboard."""
    console.print("[bold green]Starting EvalBrain Dashboard...[/bold green]")
    console.print("[yellow]Starting the API server instead...[/yellow]")
    _run_server("127.0.0.1", 8000, True)

if __name__ == "__main__":
    app()
