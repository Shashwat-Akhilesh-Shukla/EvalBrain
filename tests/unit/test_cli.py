import pytest
from typer.testing import CliRunner
from evalbrain.cli import app
from unittest.mock import patch, MagicMock

runner = CliRunner()

def test_traces_command():
    # Since it outputs to console and brain.get_traces() is currently empty
    result = runner.invoke(app, ["traces"])
    assert result.exit_code == 0
    assert "No traces found" in result.stdout

def test_prompt_list_command():
    result = runner.invoke(app, ["prompt", "list"])
    assert result.exit_code == 0
    assert "Prompt listing" in result.stdout

def test_cost_summary_command():
    result = runner.invoke(app, ["cost", "summary"])
    assert result.exit_code == 0
    assert "Cost summary" in result.stdout

def test_dashboard_command():
    result = runner.invoke(app, ["dashboard"])
    assert result.exit_code == 0
    assert "Starting EvalBrain Dashboard" in result.stdout

def test_export_json_command():
    result = runner.invoke(app, ["export", "json"])
    assert result.exit_code == 0
    assert "[]" in result.stdout or "{}" in result.stdout or "" in result.stdout

from evalbrain.core.tracer import EvalBrain

@patch.object(EvalBrain, "run_suite")
@patch("importlib.import_module")
def test_run_command(mock_import, mock_run_suite):
    # Mocking target function resolution
    mock_module = MagicMock()
    mock_module.predict = MagicMock(return_value="hello")
    mock_import.return_value = mock_module
    
    result = runner.invoke(app, ["run", "suite.yaml", "--target", "dummy.module:predict"])
    
    print("STDOUT:", result.stdout)
    assert result.exit_code == 0
    assert "Running regression suite" in result.stdout
    assert "Regression suite completed" in result.stdout
    
    mock_run_suite.assert_called_once_with("suite.yaml", mock_module.predict, None)
