import streamlit as st
import pandas as pd
import joblib

# --------------------------
# Load model + encoders
# --------------------------
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("xgb_model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

model, encoders = load_model_and_encoders()

# --------------------------
# Streamlit App UI
# --------------------------
st.title("🏠 Home Price Prediction App")
st.write("Enter property details below to estimate the **Close Price**.")

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0, value=2)
    living_area = st.number_input("Living Area (sqft)", min_value=0, value=1500)
    lot_size = st.number_input("Lot Size (sqft)", min_value=0, value=5000)
    stories = st.number_input("Stories", min_value=0, value=1)
    garage_spaces = st.number_input("Garage Spaces", min_value=0, value=2)

with col2:
    building_age = st.number_input("Building Age (years)", min_value=0, value=10)
    parking_total = st.number_input("Total Parking Spaces", min_value=0, value=2)
    main_level_bedrooms = st.number_input("Main Level Bedrooms", min_value=0, value=1)
    county = st.text_input("County", "Los Angeles")
    city = st.text_input("City", "Los Angeles")
    postal_code = st.text_input("Postal Code", "90065")

flooring = st.selectbox("Flooring", ["Wood", "Tile", "Carpet", "Mixed"])
levels = st.selectbox("Levels", ["One", "Two", "ThreeOrMore", "Unknown"])
school = st.selectbox("Expensive School District?", ["No", "Yes"])
attractions = st.number_input("Additional Attractions (score)", min_value=0, value=3)

# --------------------------
# Prepare input for prediction
# --------------------------
if st.button("Predict Price"):
    input_dict = {
        "BedroomsTotal": bedrooms,
        "BathroomsTotalInteger": bathrooms,
        "LivingArea": living_area,
        "LotSizeSquareFeet": lot_size,
        "Stories": stories,
        "GarageSpaces": garage_spaces,
        "BuildingAge": building_age,
        "ParkingTotal": parking_total,
        "MainLevelBedrooms": main_level_bedrooms,
        "CountyOrParish": county,
        "City": city,
        "PostalCode": postal_code,
        "Flooring": flooring,
        "Levels": levels,
        "ExpensiveSchoolDistrictYN": 1 if school == "Yes" else 0,
        "AdditionalAttractions": attractions,
    }

    input_df = pd.DataFrame([input_dict])

    # Apply encoders
    for col, le in encoders.items():
        try:
            input_df[col] = le.transform(input_df[col].astype(str))
        except ValueError:
            input_df[col] = -1  # unseen category

    # Ensure feature alignment
    input_df = input_df[model.get_booster().feature_names]

    try:
        prediction = model.predict(input_df)[0]
        st.subheader(f"💰 Estimated Close Price: ${prediction:,.2f}")
    except Exception as e:
        st.error("Error during prediction.")
        st.exception(e)
