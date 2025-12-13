from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def home():

    return {
        "Name": "Nwachukwu Jesse",
        "Career": "MLOps Engineering"
    }

@app.get("/health")
async def check_health():
    return {
        "RAM Status": "Nominal",
        "CPU Load": "Normal"
    }




