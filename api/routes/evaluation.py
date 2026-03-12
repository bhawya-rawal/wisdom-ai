from fastapi import APIRouter, Depends, Query
from sqlmodel import Session
from database.connection import get_session
from api.dependencies.dependencies import get_current_admin
from services.evaluation.eval_service import evaluation_service
from models.db_models import User

router = APIRouter(tags=["RAG Evaluation Panel"])

@router.post("/admin/evaluate")
def run_evaluation_benchmark(
    limit: int = Query(5, ge=1, le=100),
    admin: User = Depends(get_current_admin),
    session: Session = Depends(get_session)
):
    """Trigger a validation execution of the RAG pipeline using the 100-question benchmark dataset"""
    results = evaluation_service.run_evaluation(db_session=session, limit=limit)
    return {
        "success": True,
        "message": f"Successfully completed evaluation sweep over {limit} questions.",
        "results": results
    }

@router.get("/admin/evaluation/history")
def get_evaluation_history(
    admin: User = Depends(get_current_admin),
    session: Session = Depends(get_session)
):
    """Retrieve historical logs of completed RAGAS evaluation benchmark runs"""
    runs = evaluation_service.get_runs(db_session=session)
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_recall": r.context_recall,
            "context_precision": r.context_precision,
            "hallucination_rate": r.hallucination_rate,
            "num_questions": r.num_questions
        }
        for r in runs
    ]
