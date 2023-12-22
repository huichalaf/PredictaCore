import fastapi
import uvicorn
import os
import asyncio
import traceback

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status

from fastapi import UploadFile, Form, File
from typing import List
import pandas as pd
from io import StringIO

from neural_network import *
from keras.models import load_model

app = FastAPI()

origins = [
    "*"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST","GET","PUT","DELETE"],
    allow_headers=["*"],
)

html_files_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=html_files_path), name="static")

@app.get("/")
async def root():
    return FileResponse(html_files_path + "/index.html", media_type="text/html")

@app.post("/train")
async def train(data: UploadFile = File(...), target: UploadFile = File(...), modelName: str = Form(...)):
    try:
        data_contents = await data.read()
        target_contents = await target.read()
        data_df = pd.read_csv(StringIO(data_contents.decode()))
        target_df = pd.read_csv(StringIO(target_contents.decode()))
        X = data_df
        Y = target_df
        model = await create_model(X, Y, model_name=modelName)
        return {"status": "ok"}
    except Exception as e:
        error_trace = traceback.format_exc()
        return {"status": "error", "error": error_trace}

@app.post("/weights")
async def weights(file_name):
    path = os.path.join(os.path.dirname(__file__), "weights")
    file_path = os.path.join(path, file_name)
    return FileResponse(file_path, media_type="application/octet-stream")

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    X = data["X"]
    file_name = data["modelName"]
    X = pd.DataFrame(X[1:], columns=X[0])  # Trata la primera fila como encabezados
    X = X.apply(pd.to_numeric, errors='coerce')  # Convierte los datos a números
    print(file_name, X)
    file_path = "weights/"+file_name+"_model.h5"
    print(file_path)
    print(os.path.isfile(file_path))
    model = load_model(file_path)
    Y = await predict_(model, X)
    return JSONResponse(content={"Y": Y})


if __name__ == "__main__":
    asyncio.run(uvicorn.run(app, host="0.0.0.0", port=3000))