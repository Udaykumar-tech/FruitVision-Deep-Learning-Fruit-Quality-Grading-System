import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent))

from inference import load_all_models, predict_all_models, MODELS_INFO, DEVICE


BASE_DIR = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("FruitVision Demo — Starting up")
    print("=" * 60)
    load_all_models()
    print("=" * 60)
    print(f"Frontend served from: {FRONTEND_DIR}")
    print("Open http://localhost:8000 in your browser")
    print("=" * 60)
    yield
    print("[main] Shutting down")


app = FastAPI(
    title="FruitVision Demo API",
    description="Multi-model fruit quality classification with explainable AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    models_loaded = sum(1 for info in MODELS_INFO.values() if info["instance"] is not None)
    return {
        "status": "ok" if models_loaded == 3 else "partial",
        "device": str(DEVICE),
        "models_loaded": models_loaded,
        "total_models": 3,
    }


@app.get("/api/models")
async def get_models_info():
    return {
        "models": [
            {
                "name": name,
                "params_M": info["params_M"],
                "test_acc": info["test_acc"],
                "focus_lift": info["focus_lift"],
                "use_case": info["use_case"],
                "loaded": info["instance"] is not None,
            }
            for name, info in MODELS_INFO.items()
        ],
        "device": str(DEVICE),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Received: {file.content_type}"
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        if len(image_bytes) > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 25 MB)")

        results = predict_all_models(image_bytes)
        return JSONResponse(content=results)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


app.mount("/css",    StaticFiles(directory=str(FRONTEND_DIR / "css")),    name="css")
app.mount("/js",     StaticFiles(directory=str(FRONTEND_DIR / "js")),     name="js")
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/predict.html")
async def serve_predict():
    return FileResponse(FRONTEND_DIR / "predict.html")


@app.get("/analysis.html")
async def serve_analysis():
    return FileResponse(FRONTEND_DIR / "analysis.html")


@app.get("/methodology.html")
async def serve_methodology():
    return FileResponse(FRONTEND_DIR / "methodology.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")