from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routers import model 

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # 허용되는 origin
    allow_credentials=True, # 허용되는 HTTP Method
    allow_methods=["*"], # 허용되는 HTTP request header
    allow_headers=["*"], # 여러 origin에 대해서 쿠키가 허용되게 할 것인가 
)

app.include_router(model.router)

@app.get("/")
def read_root():
    return {"Hello" : "this is main page"}


if __name__ == "__main__":
    uvicorn.run(app="main:app",
                host="0.0.0.0",
                port=8000,
                reload=True)