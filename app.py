# app.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import RedirectResponse
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta, date, timezone
import os
from zoneinfo import ZoneInfo
# --- LangChain / Ollama imports for chatbot ---
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retriever

# --- MongoDB + auth imports ---
from db import users_col, predictions_col, chats_col, patient_history_col
from passlib.context import CryptContext
from jose import jwt, JWTError
from dotenv import load_dotenv
from bson import ObjectId
from typing import Dict, Any
from typing import List
from bson import ObjectId

from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from bson.errors import InvalidId

# -----------------------------
# Config & secrets
# -----------------------------
load_dotenv()


JWT_SECRET = os.getenv("JWT_SECRET", "change_me")
JWT_ALG = "HS256"

oauth = OAuth()

oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# -----------------------------
# FastAPI app + templates
# -----------------------------
app = FastAPI(title="Alzheimer's Prediction & Chatbot")
app.add_middleware(SessionMiddleware, secret_key=JWT_SECRET)

templates = Jinja2Templates(directory="templates")

# -----------------------------
# Load model, scaler, label encoder
# -----------------------------
MODEL_DIR = Path("model")

model_path = MODEL_DIR / "random_forest_model.pkl"
scaler_path = MODEL_DIR / "scaler.pkl"
label_path = MODEL_DIR / "label_encoder.pkl"

for p in [model_path, scaler_path, label_path]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}. Run train_model.py first. ({p})")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_path)

# -----------------------------
# Chatbot setup (Ollama + FAISS retriever)
# -----------------------------
llm = OllamaLLM(model="llama3.2")

chat_template = """
You are an assistant that explains Alzheimer's disease in clear, calm language.
Use the context below plus your own knowledge, but do NOT give medical diagnosis.
Always encourage users to consult a doctor for personal medical decisions.

Context from documents:
{data}

User question:
{question}
"""
prompt = ChatPromptTemplate.from_template(chat_template)
chat_chain = prompt | llm

# retriever based on mybuild_.pdf (from vector.py)
retriever = get_retriever()

# -----------------------------
# Auth helper functions
# -----------------------------
MAX_PASSWORD_BYTES = 72

def hash_password(password: str) -> str:
    # bcrypt can't handle > 72 bytes
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError("Password too long for bcrypt")
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(user_id: str, role: str = "user") -> str:
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=12),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except JWTError:
        return None

def get_current_user(request: Request):
    token = request.cookies.get("auth_token")
    if not token:
        return None, None
    data = decode_token(token)
    if not data:
        return None, None
    return data.get("sub"), data.get("role")



def get_admin_emails() -> set[str]:
    raw = os.getenv("ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}

def is_admin_email(email: str) -> bool:
    return email.strip().lower() in get_admin_emails()



# -----------------------------
# Patient helper functions
# -----------------------------
def calculate_age_from_birthdate(birthdate_str: str):
    """
    birthdate_str: 'YYYY-MM-DD'
    returns age in whole years (int) or None if invalid
    """
    try:
        dob = datetime.strptime(birthdate_str, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None

    today = date.today()
    age = today.year - dob.year - (
        (today.month, today.day) < (dob.month, dob.day)
    )
    return age


def get_or_create_patient_id(
    user_id: str | None,
    name: str,
    birthdate_str: str,
    gender: str,
):
    """
    If a patient with same name + birthdate + gender for this user exists,
    reuse their patient_id. Otherwise generate a new one based on birthdate.
    """
    user_obj_id = ObjectId(user_id) if user_id else None

    query = {
        "user_id": user_obj_id,
        "patient.name": name,
        "patient.birthdate": birthdate_str,
        "patient.gender": gender,
    }
    existing = predictions_col.find_one(query)
    if existing:
        patient = existing.get("patient", {}) or {}
        existing_id = patient.get("patient_id")
        if existing_id:
            return existing_id

    # No existing patient → make new ID.
    # Base: YYYYMMDD from birthdate + a small counter to avoid collisions.
    base = birthdate_str.replace("-", "")  # e.g. 20010415
    count_same_dob = predictions_col.count_documents(
        {
            "user_id": user_obj_id,
            "patient.birthdate": birthdate_str,
        }
    )
    serial = count_same_dob + 1  # 1,2,3,...
    return f"{base}-{serial}"

def to_utc_aware(dt):
    """Convert Mongo datetime to UTC-aware datetime so sorting/comparisons never crash."""
    if not dt:
        return None
    # If naive, assume it's UTC
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    # If already aware, convert to UTC
    return dt.astimezone(timezone.utc)

def require_admin(request: Request):
    user_id, role = get_current_user(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not logged in")
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    u = users_col.find_one({"_id": ObjectId(user_id)})
    if not u:
        raise HTTPException(status_code=401, detail="User not found")
    return u



# -----------------------------
# Routes
# -----------------------------

# ---- Main redirect ----
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    user_id, _ = get_current_user(request)
    if user_id:
        # already logged in -> go to dashboard
        return RedirectResponse(url="/dashboard", status_code=302)
    # not logged in -> login page
    return RedirectResponse(url="/login", status_code=302)

# ---------- DASHBOARD ----------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user_id, role = get_current_user(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user_email = None
    full_name = None

    u = users_col.find_one({"_id": ObjectId(user_id)})
    if u:
        user_email = u.get("email")
        full_name = u.get("full_name")

    # unique patient count for the prediction card
    pipeline = [
        {"$match": {"user_id": ObjectId(user_id)}},
        {"$group": {"_id": "$patient.patient_id"}}
    ]
    unique_patients = len(list(predictions_col.aggregate(pipeline)))

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user_email": user_email,
            "full_name": full_name,
            "predictions_count": unique_patients,
            "role": role,
        },
    )


# ---------- DASHBOARD CHART DATA API ----------

@app.get("/api/dashboard-charts")
async def dashboard_chart_data(request: Request):
    """
    Returns aggregated data for dashboard charts based on UNIQUE PATIENTS:
    - label_counts: how many patients in each label (using their latest prediction)
    - trend: daily patient counts for last 14 days for each label
    """
    user_id, _ = get_current_user(request)

    # ---------- 1) Get latest prediction per patient ----------
    match_filter: Dict[str, Any] = {}
    if user_id:
        match_filter["user_id"] = ObjectId(user_id)

    pipeline_latest = []
    if match_filter:
        pipeline_latest.append({"$match": match_filter})

    pipeline_latest.extend(
        [
            # newest first
            {"$sort": {"created_at": -1}},
            # keep only the latest prediction per patient
            {
                "$group": {
                    "_id": "$patient.patient_id",
                    "label": {"$first": "$label"},
                    "created_at": {"$first": "$created_at"},
                }
            },
        ]
    )

    latest_docs = list(predictions_col.aggregate(pipeline_latest))

    # helper to normalize any raw label into 3 canonical ones
    def canonical_label(raw: str) -> str:
        if raw is None:
            return "No prediction yet"
        s = str(raw).strip()

        # Alzheimer's bucket
        if s in ["1", "Alzheimer", "Alzheimer's Disease", "AD", "Alzheimer's Detected"]:
            return "Alzheimer's Detected"

        # No Alzheimer's bucket
        if "No Alzheimer" in s:
            return "No Alzheimer's"

        # Everything else = no prediction yet (manual patients etc.)
        return "No prediction yet"

    # ---------- 2) Overall label distribution (pie chart) ----------
    buckets = {
        "Alzheimer's Detected": 0,
        "No Alzheimer's": 0,
        "No prediction yet": 0,
    }

    for doc in latest_docs:
        raw_label = doc.get("label")
        canon = canonical_label(raw_label)
        buckets[canon] += 1

    # FIXED ORDER for pie chart
    label_counts = [
        {"label": "Alzheimer's Detected", "count": buckets["Alzheimer's Detected"]},
        {"label": "No Alzheimer's", "count": buckets["No Alzheimer's"]},
        {"label": "No prediction yet", "count": buckets["No prediction yet"]},
    ]

    # ---------- 3) Trend: last 14 days, unique patients per day ----------
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=13)  # 14-day window

    # Prepare day buckets
    day_labels: list[str] = []
    day_map: Dict[str, Dict[str, int]] = {}
    for i in range(14):
        d = start_date + timedelta(days=i)
        key = d.isoformat()
        day_labels.append(key)
        day_map[key] = {
            "Alzheimer's Detected": 0,
            "No Alzheimer's": 0,
        }

    for doc in latest_docs:
        created = doc.get("created_at")
        raw_label = doc.get("label")
        if not created:
            continue

        d = created.date()
        if d < start_date or d > end_date:
            continue

        key = d.isoformat()
        if key not in day_map:
            continue

        canon = canonical_label(raw_label)

        # For the trend chart we only care about true predictions,
        # ignore "No prediction yet" so it doesn't affect the lines.
        if canon in ("Alzheimer's Detected", "No Alzheimer's"):
            day_map[key][canon] += 1

    alz_series = []
    no_alz_series = []
    for key in day_labels:
        bucket = day_map[key]
        alz_series.append(bucket.get("Alzheimer's Detected", 0))
        no_alz_series.append(bucket.get("No Alzheimer's", 0))

    data = {
        "label_counts": label_counts,
        "trend": {
            "dates": day_labels,
            "alzheimers": alz_series,
            "no_alzheimers": no_alz_series,
        },
    }

    return JSONResponse(data)


# ---------- PATIENT LIST PAGE ----------

@app.get("/patients", response_class=HTMLResponse)
async def patients_page(request: Request):
    user_id, role = get_current_user(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    # Get user info (for top bar name)
    user_email = None
    full_name = None
    u = users_col.find_one({"_id": ObjectId(user_id)})
    if u:
        user_email = u.get("email")
        full_name = u.get("full_name")

    # Aggregate latest prediction per patient for THIS user
    pipeline = [
        {"$match": {"user_id": ObjectId(user_id)}},
        {"$sort": {"created_at": -1}},
        {
            "$group": {
                "_id": "$patient.patient_id",
                "name": {"$first": "$patient.name"},
                "age": {"$first": "$patient.age"},
                "gender": {"$first": "$patient.gender"},
                "phone": {"$first": "$patient.phone"},   # keep phone
                "last_label": {"$first": "$label"},
                "created_at": {"$first": "$created_at"},
            }
        },
        {"$sort": {"created_at": -1}},
    ]

    patients = []
    for row in predictions_col.aggregate(pipeline):
        patients.append(
            {
                "patient_id": row["_id"],
                "name": row.get("name"),
                "age": row.get("age"),
                "gender": row.get("gender"),
                "phone": row.get("phone"),
                "last_label": row.get("last_label"),
                "created_at": row.get("created_at"),
            }
        )

    return templates.TemplateResponse(
        "patients.html",
        {
            "request": request,
            "user_email": user_email,
            "full_name": full_name,
            "patients": patients,
            "role": role,
        },
    )

from fastapi import Form
from fastapi.responses import JSONResponse
from bson import ObjectId
from datetime import datetime

@app.post("/patients/manual")
async def add_patient_manual(
    request: Request,
    patient_name: str = Form(...),
    birthdate: str = Form(...),
    gender: str = Form(...),
    phone: str = Form(...)
):
    # who is logged in
    user_id, _ = get_current_user(request)
    if not user_id:
        return JSONResponse({"ok": False, "error": "not_logged_in"}, status_code=401)

    # age + patient_id same as prediction route
    age = calculate_age_from_birthdate(birthdate)
    patient_id = get_or_create_patient_id(user_id, patient_name, birthdate, gender)

    patient_doc = {
        "name": patient_name,
        "patient_id": patient_id,
        "birthdate": birthdate,
        "age": age,
        "gender": gender,
        "phone": phone,
    }

    # insert a "dummy" prediction row so the patient appears in your list
    doc = {
        "user_id": ObjectId(user_id),
        "patient": patient_doc,
        "inputs": None,                 # no model inputs
        "prediction_index": None,
        "predicted_class": None,
        "label": "No prediction yet",   # will show as last_label
        "created_at": datetime.utcnow(),
    }
    predictions_col.insert_one(doc)

    return JSONResponse({"ok": True, "patient_id": patient_id})



@app.get("/patients/{patient_id}/edit", response_class=HTMLResponse)
async def edit_patient_get(request: Request, patient_id: str):
    user_id, role = get_current_user(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    # latest prediction for this patient
    doc = predictions_col.find_one(
        {"user_id": ObjectId(user_id), "patient.patient_id": patient_id},
        sort=[("created_at", -1)],
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient = doc.get("patient", {}) or {}

    user_email = None
    full_name = None
    u = users_col.find_one({"_id": ObjectId(user_id)})
    if u:
        user_email = u.get("email")
        full_name = u.get("full_name")

    return templates.TemplateResponse(
        "patient_edit.html",
        {
            "request": request,
            "user_id": user_id,
            "user_email": user_email,
            "full_name": full_name,
            "role": role,
            "patient": patient,
        },
    )


@app.post("/patients/{patient_id}/edit", response_class=HTMLResponse)
async def edit_patient_post(
    request: Request,
    patient_id: str,
    name: str = Form(...),
    age: float = Form(...),
    gender: str = Form(...),
):
    user_id, _ = get_current_user(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    # update all predictions for this patient (for this user)
    predictions_col.update_many(
        {"user_id": ObjectId(user_id), "patient.patient_id": patient_id},
        {
            "$set": {
                "patient.name": name,
                "patient.age": age,
                "patient.gender": gender,
            }
        },
    )

    return RedirectResponse(url="/patients", status_code=302)

@app.post("/patients/{patient_id}/delete")
async def delete_patient(request: Request, patient_id: str):
    user_id, _ = get_current_user(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    # delete all predictions for this patient (for this user)
    predictions_col.delete_many(
        {"user_id": ObjectId(user_id), "patient.patient_id": patient_id}
    )

    return RedirectResponse(url="/patients", status_code=302)

@app.get("/add-patient", response_class=HTMLResponse)
async def add_patient_page(request: Request):
    user_id, _ = get_current_user(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    return templates.TemplateResponse(
        "add_patient.html",
        {"request": request}
    )




# ---------- PREDICTION PAGE ----------

@app.get("/prediction", response_class=HTMLResponse)
async def prediction_page(request: Request):
    user_id, role = get_current_user(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user_email = None
    u = users_col.find_one({"_id": ObjectId(user_id)})
    if u:
        user_email = u.get("email")

    # This will render your templates/Prediction.html
    return templates.TemplateResponse(
        "Prediction.html",
        {
            "request": request,
            "user_id": user_id,
            "user_email": user_email,
            "role": role,
        },
    )

@app.get("/assistant", response_class=HTMLResponse)
async def assistant_page(request: Request):
    user_id, role = get_current_user(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user_email = None
    full_name = None
    u = users_col.find_one({"_id": ObjectId(user_id)})
    if u:
        user_email = u.get("email")
        full_name = u.get("full_name")

    return templates.TemplateResponse(
        "assistant.html",
        {
            "request": request,
            "user_id": user_id,
            "user_email": user_email,
            "full_name": full_name,
            "role": role,
        },
    )


# ---------- AUTH ROUTES ----------

@app.get("/register", response_class=HTMLResponse)
async def register_get(request: Request):
    return templates.TemplateResponse(
        "register.html",
        {"request": request, "message": ""},
    )

@app.post("/register", response_class=HTMLResponse)
async def register_post(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    age: int = Form(None),
    phone: str = Form(None),
):
    # ✅ normalize email
    email_clean = email.strip().lower()

    existing = users_col.find_one({"email": email_clean})
    if existing:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "message": "Email already registered."},
        )

    try:
        hashed = hash_password(password)
    except ValueError:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "message": "Password too long. Please use a password of 72 characters or less.",
            },
        )

    # ✅ role from .env ADMIN_EMAILS
    role = "admin" if is_admin_email(email_clean) else "user"

    # ✅ always store UTC timestamp (timezone-aware)
    now_utc = datetime.now(timezone.utc)

    user = {
        "full_name": full_name.strip(),
        "email": email_clean,      # ✅ store normalized
        "password": hashed,
        "age": age,
        "phone": (phone or "").strip(),
        "role": role,              # ✅ admin if in ADMIN_EMAILS
        "status": "active",
        "created_at": now_utc,
    }

    result = users_col.insert_one(user)

    # ✅ token role must match user role
    token = create_token(str(result.inserted_id), role=role)

    # ✅ redirect admin to /admin else /dashboard
    redirect_to = "/admin" if role == "admin" else "/dashboard"
    resp = RedirectResponse(url=redirect_to, status_code=302)
    resp.set_cookie("auth_token", token, httponly=True, max_age=60 * 60 * 12)
    return resp

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "message": ""},
    )


@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    email_clean = email.strip().lower()

    # Find the user by email and check the password
    user = users_col.find_one({"email": email_clean})
    if not user or not verify_password(password, user["password"]):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "message": "Invalid email or password."}
        )

    # If the account is disabled, return the error message
    if user.get("status") == "disabled":
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "message": "Account disabled."}
        )

    # Force admin if email is in ADMIN_EMAILS (even if DB role is wrong)
    role = "admin" if is_admin_email(email_clean) else user.get("role", "user")

    # Optionally, keep the DB consistent by updating the user's role
    users_col.update_one(
        {"_id": user["_id"]},
        {"$set": {"role": role}}
    )

    # Create token for the session
    token = create_token(str(user["_id"]), role=role)

    # Redirect to admin or dashboard based on the role
    redirect_to = "/admin" if role == "admin" else "/dashboard"
    resp = RedirectResponse(url=redirect_to, status_code=302)

    # Set the authentication token in the response cookie
    resp.set_cookie("auth_token", token, httponly=True, max_age=60 * 60 * 12)  # 12 hours expiry

    return resp






@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie("auth_token")
    return resp

from datetime import datetime, timedelta, timezone
from bson import ObjectId

# ---------- ADMIN DASHBOARD ----------

# ---------- ADMIN DASHBOARD ----------
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    user_id, _ = get_current_user(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    admin_user = users_col.find_one({"_id": ObjectId(user_id)})
    if not admin_user or admin_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admins only")

    # ----- TOP STATS -----
    total_users = users_col.count_documents({})
    total_predictions = predictions_col.count_documents({})

    distinct_patient_ids = predictions_col.distinct("patient.patient_id")
    total_patients = len(distinct_patient_ids)

    # predictions created today (UTC midnight -> now)
    today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # if your DB has naive created_at (very common), this query may miss them.
    # safest approach: query using naive UTC too, and OR them.
    today_naive = today_utc.replace(tzinfo=None)

    today_filter = {
        "$or": [
            {"created_at": {"$gte": today_utc}},
            {"created_at": {"$gte": today_naive}},
        ]
    }
    today_predictions = predictions_col.count_documents(today_filter)

    # ----- PER-USER STATS -----
    pipeline_user_stats = [
        {"$match": {"user_id": {"$ne": None}}},
        {
            "$group": {
                "_id": "$user_id",
                "predictions_count": {"$sum": 1},
                "patients": {"$addToSet": "$patient.patient_id"},
                "last_activity": {"$max": "$created_at"},
            }
        },
    ]

    stats_by_user: dict = {}
    for row in predictions_col.aggregate(pipeline_user_stats):
        uid = row.get("_id")
        if not uid:
            continue
        stats_by_user[uid] = {
            "predictions_count": row.get("predictions_count", 0),
            "patients": row.get("patients", []),
            "last_activity": row.get("last_activity"),
        }

    # build list for template
    users_for_template = []
    for u in users_col.find({}):
        uid = u["_id"]
        s = stats_by_user.get(uid, {})
        patients_count = len(s.get("patients", []))
        predictions_count = s.get("predictions_count", 0)

        # normalize last_activity so sorting never breaks
        last_activity = to_utc_aware(s.get("last_activity"))

        

        users_for_template.append(
            {
                "id": str(uid),
                "full_name": u.get("full_name") or "",
                "email": u.get("email") or "",
                "role": u.get("role", "user"),
                "patients_count": patients_count,
                "predictions_count": predictions_count,
                "last_activity": last_activity,
            }
        )

    # sort users by last_activity desc (timezone-safe)
    users_for_template.sort(
        key=lambda x: x["last_activity"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    return templates.TemplateResponse(
        "admin_dashboard.html",
        {
            "request": request,
            "full_name": admin_user.get("full_name"),
            "user_email": admin_user.get("email"),
            "total_users": total_users,
            "total_patients": total_patients,
            "total_predictions": total_predictions,
            "today_predictions": today_predictions,
            "users": users_for_template,
        },
    )

# ---------- ADMIN: USERS LIST ----------
@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users(request: Request):
    admin_user = require_admin(request)  # should return the admin's user doc

    # --- stats per user from predictions ---
    pipeline = [
        {"$match": {"user_id": {"$ne": None}}},
        {
            "$group": {
                "_id": "$user_id",
                "predictions_count": {"$sum": 1},
                "patients": {"$addToSet": "$patient.patient_id"},
                "last_activity": {"$max": "$created_at"},
                "first_activity": {"$min": "$created_at"},
            }
        },
    ]

    stats_by_user = {row["_id"]: row for row in predictions_col.aggregate(pipeline)}

    now_utc = datetime.now(timezone.utc)

    def to_utc_aware(dt):
        if not dt:
            return None
        # Mongo may return naive datetime; normalize it
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    users_for_template = []
    for u in users_col.find({}):
        uid = u["_id"]
        s = stats_by_user.get(uid, {})

        created_at = to_utc_aware(u.get("created_at"))
        first_activity = to_utc_aware(s.get("first_activity"))
        last_activity = to_utc_aware(s.get("last_activity"))

        start_date = created_at or first_activity
        days_using = (now_utc - start_date).days if start_date else 0

        patients_list = s.get("patients", []) or []

        users_for_template.append({
            "id": str(uid),
            "full_name": u.get("full_name") or "",
            "email": u.get("email") or "",
            "phone": u.get("phone") or "",
            "role": u.get("role", "user"),
            "status": u.get("status", "active"),
            "days_using": days_using,
            "patients_count": len(patients_list),
            "predictions_count": s.get("predictions_count", 0),
            "last_activity": last_activity,
        })

    users_for_template.sort(
        key=lambda x: x["last_activity"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True
    )

    return templates.TemplateResponse(
        "admin_users.html",
        {
            "request": request,
            "full_name": admin_user.get("full_name"),
            "user_email": admin_user.get("email"),
            "users": users_for_template,
        },
    )
def build_patient_history(user_id: str, patient_id: str):
    uid = ObjectId(user_id)
    history = []

    # 1) Predictions history
    preds = list(
        predictions_col.find(
            {"user_id": uid, "patient.patient_id": patient_id},
            sort=[("created_at", -1)],
        )
    )

    for p in preds:
        inputs = p.get("inputs") or {}
        history.append({
            "kind": "prediction",
            "created_at": to_utc_aware(p.get("created_at")),
            "label": p.get("label"),
            "details": {
                "memory_complaints": inputs.get("memory_complaints"),
                "behavioral_problems": inputs.get("behavioral_problems"),
                "adl": inputs.get("adl"),
                "mmse": inputs.get("mmse"),
                "functional_assessment": inputs.get("functional_assessment"),
            },
        })

    # 2) Notes history (NEW)
    notes = list(
        patient_history_col.find(
            {"user_id": uid, "patient_id": patient_id},
            sort=[("created_at", -1)],
        )
    )

    for n in notes:
        history.append({
            "kind": "note",
            "created_at": to_utc_aware(n.get("created_at")),
            "label": "Note",
            "details": n.get("details"),
        })

    # 3) Sort mixed history
    history.sort(
        key=lambda x: x.get("created_at") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return history


def add_patient_note(user_id: str, patient_id: str, details: str):
    details = (details or "").strip()
    if not details:
        return False

    patient_history_col.insert_one({
        "user_id": ObjectId(user_id),
        "patient_id": patient_id,
        "details": details,
        "created_at": datetime.utcnow(),
    })
    return True
@app.get("/auth/google/login")
async def google_login(request: Request):
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/google/callback")
async def google_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)

    userinfo = token.get("userinfo")
    if not userinfo:
        userinfo = await oauth.google.parse_id_token(request, token)

    email = (userinfo.get("email") or "").strip().lower()
    full_name = (userinfo.get("name") or "Google User").strip()

    if not email:
        raise HTTPException(status_code=400, detail="Google did not return email")

    user = users_col.find_one({"email": email})

    # ✅ IF USER EXISTS → SHOW MESSAGE ON REGISTER PAGE
    if user:
        return RedirectResponse(
            url="/register?msg=already_registered",
            status_code=302
        )

    # ✅ NEW USER → CREATE ACCOUNT
    role = "admin" if is_admin_email(email) else "user"
    now_utc = datetime.now(timezone.utc)

    res = users_col.insert_one(
        {
            "full_name": full_name,
            "email": email,
            "password": None,
            "age": None,
            "phone": "",
            "role": role,
            "status": "active",
            "created_at": now_utc,
            "auth_provider": "google",
        }
    )

    user_id = str(res.inserted_id)
    jwt_token = create_token(user_id, role)

    redirect_to = "/admin" if role == "admin" else "/dashboard"
    resp = RedirectResponse(url=redirect_to, status_code=302)
    resp.set_cookie("auth_token", jwt_token, httponly=True, max_age=60 * 60 * 12)
    return resp


# ---------- ADMIN: TOGGLE ENABLE/DISABLE ----------
@app.post("/admin/users/{user_id}/toggle")
async def admin_toggle_user(request: Request, user_id: str):
    admin_user = require_admin(request)

    try:
        target_id = ObjectId(user_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid user id")

    # Prevent disabling yourself
    if str(admin_user["_id"]) == str(target_id):
        raise HTTPException(status_code=400, detail="You cannot disable your own account")

    u = users_col.find_one({"_id": target_id})
    if not u:
        raise HTTPException(status_code=404, detail="User not found")

    new_status = "disabled" if u.get("status", "active") == "active" else "active"
    users_col.update_one(
        {"_id": target_id},
        {"$set": {"status": new_status}}
    )

    return RedirectResponse(url="/admin/users", status_code=303)

# ---------- ADMIN: DELETE USER ----------
@app.post("/admin/users/{user_id}/delete")
async def admin_delete_user(request: Request, user_id: str):
    admin_user = require_admin(request)

    try:
        target_id = ObjectId(user_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid user id")

    # Prevent deleting yourself
    if str(admin_user["_id"]) == str(target_id):
        raise HTTPException(status_code=400, detail="You cannot delete your own account")

    # delete user + their data
    users_col.delete_one({"_id": target_id})
    predictions_col.delete_many({"user_id": target_id})
    chats_col.delete_many({"user_id": target_id})

    return RedirectResponse(url="/admin/users", status_code=303)

# ---------- ADMIN: EDIT USER ----------
@app.post("/admin/users/{user_id}/edit")
async def admin_edit_user(
    request: Request,
    user_id: str,
    full_name: str = Form(""),
    phone: str = Form(""),
    role: str = Form("user"),
):
    admin_user = require_admin(request)

    try:
        target_id = ObjectId(user_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid user id")

    # Optional: prevent admin from removing their own admin role
    if str(admin_user["_id"]) == str(target_id) and role != "admin":
        raise HTTPException(status_code=400, detail="You cannot remove your own admin role")

    role = "admin" if role == "admin" else "user"

    users_col.update_one(
        {"_id": target_id},
        {"$set": {"full_name": full_name.strip(), "phone": (phone or "").strip(), "role": role}},
    )
    return RedirectResponse(url="/admin/users", status_code=303)


# ---------- ADMIN CHART DATA API ----------

@app.get("/api/admin-charts")
async def admin_chart_data(request: Request):
    """
    Global (admin) chart data:
    - label_counts: number of patients in each label, using latest prediction
    - trend: last 30 days, unique patients per label by their latest prediction
    """

    # ----- 1) latest prediction per patient (all users) -----
    pipeline_latest = [
        {"$sort": {"created_at": -1}},  # newest first
        {
            "$group": {
                "_id": "$patient.patient_id",
                "label": {"$first": "$label"},
                "created_at": {"$first": "$created_at"},
            }
        },
    ]

    latest_docs = list(predictions_col.aggregate(pipeline_latest))

    # ----- 2) overall label distribution -----
    counts_map: dict[str, int] = {}
    for doc in latest_docs:
        label = doc.get("label") or "Unknown"
        counts_map[label] = counts_map.get(label, 0) + 1

    # if you also store "manual" patients with no predictions
    # in a separate collection, you could add a "No prediction yet"
    # bucket here by joining with that collection.

    label_counts = [
        {"label": label, "count": count} for label, count in counts_map.items()
    ]

    # ----- 3) Trend: last 30 days, unique patients per day -----
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=29)

    day_labels: list[str] = []
    day_map: dict[str, dict[str, int]] = {}

    for i in range(30):
        d = start_date + timedelta(days=i)
        key = d.isoformat()
        day_labels.append(key)
        # we initialise bucket but we allow any label later
        day_map[key] = {}

    for doc in latest_docs:
        created = doc.get("created_at")
        label = doc.get("label", "Unknown")
        if not created:
            continue

        d = created.date()
        if d < start_date or d > end_date:
            continue

        key = d.isoformat()
        bucket = day_map.get(key)
        if bucket is None:
            continue

        bucket[label] = bucket.get(label, 0) + 1

    # For line chart we still expose only two series:
    # "Alzheimer's Detected" vs "No Alzheimer's".
    alz_series = []
    no_alz_series = []

    for key in day_labels:
        bucket = day_map[key]
        alz_series.append(bucket.get("Alzheimer's Detected", 0))
        # everything that is not "Alzheimer's Detected" we can
        # treat as "No Alzheimer's" for this chart
        no_alz = bucket.get("No Alzheimer's", 0)
        # optionally include "No prediction yet" etc. into no_alz or separate line
        no_alz_series.append(no_alz)

    data = {
        "label_counts": label_counts,
        "trend": {
            "dates": day_labels,
            "alzheimers": alz_series,
            "no_alzheimers": no_alz_series,
        },
    }

    return JSONResponse(data)




# ---------- PREDICTION ENDPOINT ----------

@app.post("/predict")
async def predict_api(
    request: Request,
    # patient info
    patient_name: str = Form(...),
    birthdate: str = Form(...),  # 'YYYY-MM-DD'
    gender: str = Form(...),
    phone: str = Form(None),     # <<--- NEW
    # model features
    memory_complaints: float = Form(...),
    behavioral_problems: float = Form(...),
    adl: float = Form(...),
    mmse: float = Form(...),
    functional_assessment: float = Form(...),
):
    features = np.array(
        [[memory_complaints, behavioral_problems, adl, mmse, functional_assessment]],
        dtype=float,
    )
    scaled = scaler.transform(features)

    pred_idx = int(model.predict(scaled)[0])
    class_label = label_encoder.inverse_transform([pred_idx])[0]

    if str(class_label) in ["1", "Alzheimer", "Alzheimer's Disease", "AD"]:
        status = "Alzheimer's Detected"
    else:
        status = "No Alzheimer's"

    user_id, _ = get_current_user(request)
    age = calculate_age_from_birthdate(birthdate)
    patient_id = get_or_create_patient_id(user_id, patient_name, birthdate, gender)

    # store phone here
    patient_doc = {
        "name": patient_name,
        "patient_id": patient_id,
        "birthdate": birthdate,
        "age": age,
        "gender": gender,
        "phone": phone,
    }

    doc = {
        "user_id": ObjectId(user_id) if user_id else None,
        "patient": patient_doc,
        "inputs": {
            "memory_complaints": memory_complaints,
            "behavioral_problems": behavioral_problems,
            "adl": adl,
            "mmse": mmse,
            "functional_assessment": functional_assessment,
        },
        "prediction_index": pred_idx,
        "predicted_class": str(class_label),
        "label": status,
        "created_at": datetime.utcnow(),
    }
    predictions_col.insert_one(doc)

    return JSONResponse(
        {
            "prediction_index": pred_idx,
            "predicted_class": str(class_label),
            "label": status,
            "patient_id": patient_id,
        }
    )


# ---------- CHATBOT ENDPOINT ----------

@app.post("/chat")
async def chat_api(request: Request, question: str = Form(...)):
    """
    Receives a user question, retrieves relevant chunks from the PDF via FAISS,
    sends them + question to Ollama, and returns the answer.
    """
    try:
        docs = retriever.invoke(question)
        if docs:
            data_text = "\n\n".join(d.page_content for d in docs)
        else:
            data_text = "General information about Alzheimer's disease."

        answer = chat_chain.invoke({"data": data_text, "question": question})

        # save chat history in MongoDB
        user_id, _ = get_current_user(request)
        chats_col.insert_one(
            {
                "user_id": ObjectId(user_id) if user_id else None,
                "question": question,
                "answer": str(answer),
                "created_at": datetime.utcnow(),
            }
        )

        return JSONResponse({"answer": str(answer)})
    except Exception as e:
        return JSONResponse({"answer": f"⚠️ Error: {e}"}, status_code=500)
