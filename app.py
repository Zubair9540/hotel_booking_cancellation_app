import joblib
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="centered",
)

# Model Load karna
model = joblib.load("gb_booking_model.pkl")

# Yahan apne image ka naam likh dein jo folder me save hai (jaise 'hotel.jpg')
st.image("z.jpg", width="stretch")

st.title("Hotel Booking Cancellation Predictor")
st.divider()
st.write("Enter booking details to predict cancellation.")

lead_time = st.slider("Lead Time", 0, 500, 50)
avg_price = st.number_input("Average Price Per Room", 0.0, 500.0, 100.0)
special_requests = st.slider("Number of Special Requests", 0, 5, 1)
total_guests = st.slider("Total Guests", 1, 10, 2)
total_nights = st.slider("Total Nights", 1, 20, 3)
repeated_guest = st.selectbox("Repeated Guest", [0, 1])

if st.button("Predict"):
  input_data = np.array([[
      lead_time,
      avg_price,
      special_requests,
      total_guests,
      total_nights,
      repeated_guest,
  ]])

  prediction = model.predict(input_data)[0]
  prob = model.predict_proba(input_data)[0][1]

  if prediction == 1 or prediction == "Canceled":
    st.error(f"Prediction: Canceled (Probability: {prob:.2f})")
  else:
    st.success(f"Prediction: Not Canceled (Probability: {1-prob:.2f})")