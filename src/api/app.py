from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials

# Routers
from auth.routes import router as auth_router
# from routes.upload_routes import router as upload_router   ← You'll add this soon

# Logger (defined in your utils module)
from utils.logging import get_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)
logger.info("Starting FastAPI App")

# Firebase Admin SDK init (once globally)
try:
    cred = credentials.Certificate("firebase-adminsdk.json")  # Path to your service account key
    firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin initialized")
except Exception as e:
    logger.error(f"Failed to initialize Firebase Admin: {e}")
    raise RuntimeError("Firebase initialization failed. Check credentials.")

# Init FastAPI app
app = FastAPI(
    title="Auto EDA Backend",
    description="Backend for the Auto EDA LangGraph App",
    version="1.0.0"
)

# CORS Middleware — allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Include routers
app.include_router(auth_router)
# app.include_router(upload_router)  ← You'll enable this later


# Health check
@app.get("/")
def root():
    return {"message": "🚀 Auto EDA backend is live!"}
