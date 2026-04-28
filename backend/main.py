from fastapi import FastAPI, HTTPException, Depends, File, Form, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from slugify import slugify
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from backend.models import ResearchRequest, ResearchResponse
from backend.research_chain import run_full_research
from backend.database import (
    get_db,
    KnowledgeBase,
    ChatHistory,
    User,
    init_db,
)
from backend.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_access_token,
)
from backend.rag.config import settings as rag_settings
from backend.rag.document_processor import load_document_bytes, document_to_chunks
from backend.rag.qa import answer_question
from backend.rag.vector_store import VectorStore

load_dotenv()
init_db()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Energy Intelligence Engine", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# Response Models
# ==================================================

class UploadResponse(BaseModel):
    project: str
    file_names: List[str]
    chunk_count: int
    message: str


class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    retrieval_count: int


# ==================================================
# Utility Functions
# ==================================================

def normalize_topic(topic: str) -> str:
    return slugify(topic.strip())


def fetch_from_cache(topic: str, db: Session):
    slug = normalize_topic(topic)
    return db.query(KnowledgeBase).filter(KnowledgeBase.slug == slug).first()


def store_in_cache(topic: str, report: str, db: Session):
    slug = normalize_topic(topic)
    existing = db.query(KnowledgeBase).filter(KnowledgeBase.slug == slug).first()

    if not existing:
        db.add(KnowledgeBase(query=topic, slug=slug, content=report))
        db.commit()


def save_to_history(topic: str, report: str, db: Session):
    db.add(ChatHistory(query=topic, response=report))
    db.commit()


def archive_report(topic: str, report: str) -> Optional[str]:
    try:
        folder = os.path.join(os.path.dirname(__file__), "knowledge_base")
        os.makedirs(folder, exist_ok=True)

        filename = f"{normalize_topic(topic)}.txt"
        full_path = os.path.join(folder, filename)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(f"Topic: {topic}\nGenerated At: {datetime.utcnow()}\n\n{report}")

        return full_path

    except Exception:
        return None


# ==================================================
# Authentication Helpers
# ==================================================

def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing token")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")

    token = authorization.replace("Bearer ", "")
    payload = decode_access_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    username = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# ==================================================
# Startup
# ==================================================

@app.on_event("startup")
def startup_event():
    Path(rag_settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
    logger.info("Chroma storage ready")


# ==================================================
# Auth Routes
# ==================================================

@app.post("/signup")
def signup(data: dict, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == data["username"]).first()

    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user = User(
        username=data["username"],
        email=data["email"],
        hashed_password=get_password_hash(data["password"])
    )

    db.add(user)
    db.commit()

    token = create_access_token({"sub": user.username})

    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username
    }


@app.post("/token")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.username})

    return {
        "access_token": token,
        "token_type": "bearer"
    }


@app.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "email": current_user.email
    }


# ==================================================
# Existing Routes
# ==================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    project: str = Form(None),
):
    project_name = project or rag_settings.default_project

    try:
        data = await file.read()

        if not data:
            raise HTTPException(status_code=400, detail="Empty file")

        source_name, text = load_document_bytes(file.filename, data)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found")

        chunks = document_to_chunks(source_name, text)

        store = VectorStore(project_name)
        store.add_documents(chunks)

        return UploadResponse(
            project=project_name,
            file_names=[file.filename],
            chunk_count=len(chunks),
            message="File indexed successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(
    question: str = Form(...),
    project: str = Form(None),
    top_k: int = Form(5),
):
    try:
        project_name = project or rag_settings.default_project
        payload = answer_question(question, project=project_name, top_k=top_k)
        return RAGQueryResponse(**payload)

    except Exception as e:
      traceback.print_exc()   # 🔥 this will show exact line
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/research", response_model=ResearchResponse)
async def research_controller(
    payload: ResearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        cached = fetch_from_cache(payload.query, db)

        if cached:
            save_to_history(payload.query, cached.content, db)

            return ResearchResponse(
                query=payload.query,
                result=cached.content,
                file_path="database-cache",
                suggestions=[]
            )

        output = run_full_research(
            topic=payload.query,
            thread_id=payload.thread_id,
            rag_project=rag_settings.default_project,
        )

        report = output.get("report")
        suggestions = output.get("suggestions", [])

        store_in_cache(payload.query, report, db)
        save_to_history(payload.query, report, db)

        file_path = archive_report(payload.query, report)

        return ResearchResponse(
            query=payload.query,
            result=report,
            file_path=file_path,
            suggestions=suggestions
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=List[ResearchResponse])
async def recent_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logs = db.query(ChatHistory).order_by(ChatHistory.created_at.desc()).limit(5).all()

    return [
        ResearchResponse(
            query=item.query,
            result=item.response,
            file_path=None,
            suggestions=[]
        )
        for item in logs
    ]


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "3.0.0"
    }