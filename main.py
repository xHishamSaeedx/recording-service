from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from route import router as service_router
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include the API router from route.py
app.include_router(service_router)

# Root endpoint for basic connectivity check
@app.get("/")
def read_root():
    return {"message": "Hello, from uploaded recording processing service!"}

# Run the app using Uvicorn for ASGI support, enabling WebSocket functionality
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=True)
