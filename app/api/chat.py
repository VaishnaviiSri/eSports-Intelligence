"""Chat API route for grounded rulebook answers."""

from fastapi import APIRouter, HTTPException

from app.qa.answerer import IndexUnavailableError, answer_question
from app.qa.llm_client import LLMUnavailableError
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Return a grounded answer or a controlled service error."""
    try:
        return answer_question(request)
    except IndexUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMUnavailableError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Groq request failed: {str(e)}",
        ) from e