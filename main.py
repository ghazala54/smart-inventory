from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = FastAPI(title="House Price Prediction API")

# السماح للواجهة الأمامية بالتواصل مع الـ Backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# تعريف نمط البيانات المدخلة القادمة من Frontend
class HouseFeatures(BaseModel):
  MedInc: float
  HouseAge: float
  AveRooms: float
  AveBedrms: float
  Population: float
  AveOccup: float
  Latitude: float
  Longitude: float


# تدريب النموذج عند بدء تشغيل الـ Backend
housing = fetch_california_housing(as_frame=True)
X = housing.frame.drop(columns=["MedHouseVal"])
y = housing.frame["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)


@app.get("/")
def home():
  return {"message": "House Price Prediction API is running!"}


@app.post("/predict")
def predict_price(data: HouseFeatures):
  input_data = [[
      data.MedInc,
      data.HouseAge,
      data.AveRooms,
      data.AveBedrms,
      data.Population,
      data.AveOccup,
      data.Latitude,
      data.Longitude,
  ]]
  prediction = model.predict(input_data)[0]
  predicted_price_dollars = round(prediction * 100000, 2)

  return {"predicted_price": predicted_price_dollars}