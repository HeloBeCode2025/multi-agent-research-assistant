import os
import time
import mlflow
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
mlflow.set_experiment("research-assistant")

def track_agent_run(question: str, app):
    """Run the agent graph and log everything to MLflow."""
    with mlflow.start_run():
        mlflow.log_param("question", question[:250])
        mlflow.log_param("provider", os.getenv("LLM_PROVIDER"))
        mlflow.log_param("model", os.getenv("OLLAMA_MODEL", "unknown"))
        
        start = time.time()
        result = app.invoke({"question": question})
        latency = time.time() - start
        
        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_metric("revision_count", result["revision_count"])
        mlflow.log_metric("draft_length", len(result["draft"]))
        mlflow.log_metric("is_approved", int(result["is_approved"]))
        
        # Log the final draft as an artifact
        with open("/tmp/draft.txt", "w") as f:
            f.write(result["draft"])
        mlflow.log_artifact("/tmp/draft.txt")
        
        return result