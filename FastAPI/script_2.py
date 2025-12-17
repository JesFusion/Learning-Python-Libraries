from fastapi import FastAPI, Body
import requests
import json
import logging
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import requests
from dotenv import load_dotenv
import logging




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
















































































app = FastAPI()

# ===================================== Path vs. Query Parameters =====================================

'''
Path Parameters points to a specific resource 
For example: "/items/model_v5"
- Here we're trying to access the 5th version of our model

Query Parameters modifies how you view the resource. It comes after "?"

For Example: "/predict?user_ID=99&threshold=0.8"
- Here, we're trying to access the prediction, but only if the user_ID is valid and the confidence is > 80%
'''

@app.get("/")
async def home_page():
    return "Welcome to FastAPI!"


@app.get("/predict/{model}")
async def churn_predictor(

    model: str, # Path Parameter (matched in {}) 
    user_ID: int, # Query Parameter (not in {})
    treshold: float = 0.1, # Query Parameter with Default

): # FastAPI performs type checking befofe passing values. If you passed a string to user_ID, it'll throw an error
    
    output = {
        "Instantiated Model": model,
        "User ID": "ID_" + str(user_ID),
        "Confidence Threshold": treshold,
        "Model Prediction": {
            "Churn Risk": "High",
            "Model Performance": "threshold"
        }
    }

    return output
    # try running "/predict/v1?user_ID=88&threshold=0.3"



























































# =====================================  =====================================
# =====================================  =====================================



app = FastAPI()

@app.get("/")
async def home_page():

    home_output = {
        "Name": "Jesse",
        "is Cool": True
    }

    return home_output


@app.post("/ingest/raw_data")
async def ingest_data(payload_information: dict = Body(...)):

    """
    Simulates an endpoint that accepts raw data (The Moving Truck).
    
    Args:
        payload (dict): This tells FastAPI to expect a JSON body, 
                        and to convert (deserialize) it into a Python dictionary.
                        'Body(...)' means the body is REQUIRED.
    """

    the_ID = payload_information.get("ID")

    the_price = payload_information.get("price")

    output = f"Received! From what I have here, your ID is {the_ID} and you're willing to pay ${the_price} for the house"

    return output




# ===================================== Section 2 =====================================




logging.basicConfig(
    format = "--> %(message)s",
    level = logging.DEBUG
)


the_url = "http://127.0.0.1:8000/ingest/raw_data"

sample_data = {
    "ID": "id_12910",

    "address": "14 Agada Street Co-operative Housing Abakpa Nike Enugu",

    "sqft": 5698,

    "price": 1700000,

    "features": ['garage', 'pool', 'master bedroom', 'solar panels', 'gym']
}

the_payload = json.dumps(sample_data, indent = 3) # json.dumps means "dump to string" (also called Serialization). Normally if you see the output, it'll be one long line of string, but we specified that 3 lines of indentation be added to visual clarity (indent = 3)

logging.info(f'''
Sending Payload: {the_payload}
''')

server_response = requests.post(the_url, json = sample_data) # We use requests.post() over the API. The 'json=' argument handles Serialization for us automatically.


logging.info(f'''
======================================== Server Response ========================================
             
{server_response.json()}
''')


# =====================================  =====================================






















































# =====================================  =====================================
# =====================================  =====================================




app = FastAPI()


class InputDataFormat(BaseModel):

    ID: int

    price: float = Field(..., gt = 0, description = "Price of House in USD") # gt means "greater than", we're trying to specify that he price amout coming in shouldn't be greater than 0

    sqft: int

    features: List[str] # features MUST be a list, and every item in that list MUST be a string

    address: Optional[str] = "Homeless" # Setting 'Optional' means that it's optional to fill this section. If it's not filled, the default value (Homeless) is passed


# loading app...

@app.post("/data_validation")
async def house_validation(HOUSE: InputDataFormat):

    """
    This function ONLY executes if the data passes the Pydantic checks.
    """

    # We now have 'Dot Notation'.

    # We don't need to write payload.get('price'). We can use "payload.price"

    # This enables auto-complete in VS Code and prevents typos

    house_id = f"h_{HOUSE.ID}"

    each_feature_price = HOUSE.price / len(HOUSE.features)

    output = f"House {house_id}: House Address = {HOUSE.address}, House price = ${HOUSE.price}, Price per feature = {each_feature_price}"

    return output










logging.basicConfig(
    level = logging.DEBUG,
    format = "%(levelname)s :: %(message)s"
)

load_dotenv()

the_url = f"{os.getenv("HOME_URL")}data_validation"

# let's simulate some bad data to see if Pydantic works...
some_stupid_data = {
    "ID": "open",
    "price": -35.23,
    "features": "We're sending features as a string not a list"
}

# Sending trash data to API
api_response = requests.post(the_url, json = some_stupid_data)

# Pydantic should throw an error
logging.error(f"{api_response.status_code}")

logging.error(f'''
======================================== Pydantic Error Message ========================================

{api_response.json()}
''')


# =====================================  =====================================




