from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from infer import ask_question

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    game_over: bool = False

@app.post("/ask", response_model=AnswerResponse)
async def ask_movie_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        answer = ask_question(request.question)
        resp_lower = answer.lower()
        game_over = resp_lower.startswith("yes") and "correct" in resp_lower
        return {"answer": answer, "game_over": game_over}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
