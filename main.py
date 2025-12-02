from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from lstm_model import train_and_predict
import akshare as ak
import pandas as pd

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def predict(request: Request, symbol: str = Form(...)):
    symbol = symbol.strip().upper()
    try:
        data = train_and_predict(symbol=symbol, epochs=50)
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"获取数据失败：{str(e)}（请确认代码正确，如 RB2410、AU2412、M2501 等）"
        })
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "data": data,
        "symbol": symbol
    })

if __name__ == "__main__":
    import uvicorn
    os.makedirs("models", exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)