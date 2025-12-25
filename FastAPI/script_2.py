import os
import requests
import json
import joblib
import logging
import numpy as np
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from fastapi import FastAPI, Body
from typing import List, Optional
from dotenv import load_dotenv


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



























































# ===================================== API =====================================



app = FastAPI()

@app.get("/")
async def main():

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





# ===================================== Client =====================================



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

























































# ===================================== API =====================================




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





# ===================================== Client =====================================



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












































































# ===================================== API =====================================


"""
Nested Models & Complex Structures:

In the enterprise world (and in JSON), data is often deeply nested—like a Russian Doll or a file folder system.

Imagine you aren't just predicting the price of one house. You are Zillow. You need to predict prices for an entire neighborhood at once (Batch Inference). Or, a house object might contain an Address object inside it, which contains a GeoLocation object inside that.

Composition (The "Lego" Approach): Instead of writing one giant validation list, we build small, reusable models (Address, Owner, Features) and plug them into a main model (House).

The Code: We are going to build a Batch Prediction Endpoint. This is standard MLOps practice because running your model once on 50 rows is way faster (Vectorization!) than running it 50 times on 1 row.
"""


belarus = FastAPI()

# ===================================== SUB-MODELS =====================================
class UserAddress(BaseModel):

    street_location: str
    city_of_origin: str
    current_zip_code: str

class ObservableHouseFeatures(BaseModel):

    square_feet: int
    no_of_rooms: int
    pool_present: bool = False # Default should be false

# ===================================== PARENT MODEL =====================================

class ListingOfTheHouse(BaseModel):

    the_address: UserAddress

    present_features: ObservableHouseFeatures

    sales_agent: Optional[str] = None


# ===================================== BATCH MODEL =====================================

class RequestOfTheBatch(BaseModel):

    id_of_batch: str

    the_items: List[ListingOfTheHouse]



# ===================================== ENDPOINTS =====================================

@belarus.get("/")
def is_cools():

    return "Jesse is cool"


@belarus.post("/batch_process")
async def batch_processing(package: RequestOfTheBatch):

    """
    Receives a batch of houses.
    FastAPI will loop through EVERY house in the list and validate it 
    against the HouseListing schema, the Address schema, etc.
    If even ONE field in ONE house is wrong, the whole batch is rejected.
    """

    container_list = []

    for a_house in package.the_items:

        assumed_price = (a_house.present_features.square_feet) * 219

        container_list.append({
            "house_zip": a_house.the_address.current_zip_code,

            "model_prediction": assumed_price
        })

    output = {
        "id_of_batch": package.id_of_batch,

        "number_of_processed_items": len(container_list),

        "the_results": container_list
    }

    return output




    

# ===================================== Client =====================================


load_dotenv()

input_data = {
  "id_of_batch": "batch_001",

  "the_items": [
    {
      "the_address": {
           "street_location": "123 ML St",
           
           "city_of_origin": "Lagos",
           
           "current_zip_code": "100001" 
      },

      "present_features": {
           "square_feet": 1500,
           
           "no_of_rooms": 3,
           
           "pool_present": True 
      }
    },


    {
      "the_address": {
           "street_location": "456 AI Ave",
           
           "city_of_origin": "Abuja",
           
           "current_zip_code": "900001" 
      },

      "present_features": {
          "square_feet": 3000, 
          
          "no_of_rooms": 5
      },

      'sales_agent': "Njoku Kingsley Anthony"
    }
  ]
}


server_response = requests.post(
    json = input_data,

    url = f"{os.getenv("HOME_URL")}batch_process"
)


print(f'''
Server Output: {server_response.json()}

Status Code: {server_response.status_code}
''')

























































# ===================================== API =====================================



load_dotenv()

# model_save_path = os.getenv("MODEL_SAVE_PATH")
model_save_path = '/home/jesfusion/Documents/ml/ML-Learning-Repository/Saved_Datasets_and_Models/Models/'

# loading the model and scaler
diamond_model = joblib.load(f"{model_save_path}KNN/diamond_model.pkl")

diamond_scaler = joblib.load(f"{model_save_path}KNN/diamond_scaler.pkl")

diamond_api = FastAPI()

# defining input data schema...

class CutType(str, Enum):
    FAIR = "fair"
    GOOD = "good"
    IDEAL = "ideal"


class DiamondInputSchema(BaseModel):

    carat: float

    cut: CutType # cut must be either fair, good or ideal


# defining our endpoint and it's function
@diamond_api.post("/predict/price")
async def diamond_price_prediction(diamond: DiamondInputSchema):

    c_string_conversion = {
        "fair": 0,
        "good": 1,
        "ideal": 2
    }

    carat_weight = diamond.carat

    # we'll map cut weight to it's value
    cut_number = c_string_conversion.get(diamond.cut)

    # convert to an array and pass it to our scaler, who can put it in the language our model understands
    input_val_array = np.array([[carat_weight, cut_number]])

    model_feed = diamond_scaler.transform(input_val_array)

    # collecting model answer and passing back to client
    model_answer = float(diamond_model.predict(model_feed)[0])

    return model_answer






# ===================================== Client =====================================

load_dotenv()

# let's simulate a user that wants to make use of our API
carat_weight = input("Enter a Carat Weight: ")

c_rating = input("Enter a Carat Rating: ").lower() # all letters in c_rating will be converted to lowercase, to prevent pydantic from throwing an error

user_input = {
    'carat': carat_weight, # carat_weight is converted to a float, because it's pydantic expects

    'cut': c_rating
}

# sending user input to model through the API...
d_pred_api_response = requests.post(url = f"{os.getenv("HOME_URL")}predict/price", json = user_input).json()

# printing out AI Response
print(f"CrystalClear AI Predicts that diamond is worth ${d_pred_api_response}")














































































# ===================================== API =====================================




API = FastAPI()

class FeaturesOfHouse(BaseModel):

    square_feet: float = Field(gt = 0, description = "Total Area of House")

    number_of_rooms: int = Field(gt = 0, description = "Number of rooms in the House")

    @model_validator(mode = "after") # this runs after incoming data has been allowed to passthrough by the pydantic bodyguard
    def room_check_density(self):

        # in this function, we throw an error if the square feet of the incoming house is unreasonable

        house_area = self.square_feet

        house_rooms = self.number_of_rooms

        if (house_area / house_rooms) < 50:

            error = f"Impossible Data: {house_rooms} rooms cannot fit in {house_area} square feet."

            raise ValueError(error)
        
        return self


@API.post("/predict/house")
async def predict_house_price(the_house: FeaturesOfHouse):

    price = (the_house.square_feet ** 2) / (the_house.number_of_rooms * 58.257)

    output = {
        "status": "valid",

        "density": the_house.square_feet / the_house.number_of_rooms,

        "price": f"${price:.2f}"
    }

    return output





# ===================================== Client =====================================


load_dotenv()

user_input = {
    'square_feet': 7500,

    'number_of_rooms': 4
}

# sending user input to model through the API...
d_pred_api_response = requests.post(url = f"{os.getenv("HOME_URL")}predict/house", json = user_input).json()

# printing out AI Response
print(f"API Response: {d_pred_api_response}")






