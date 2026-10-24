import streamlit as st

st.set_page_config(page_title="Drowsiness Detection", page_icon="🚗")

st.title("🚗 Driver Drowsiness Detection")
st.write("Real-Time Drowsiness and Yawning Detection System")

st.info("This is the project demo page.")

st.subheader("Project Features")
st.write("👁️ Eye closure detection")
st.write("🥱 Yawning detection")
st.write("📉 Head drooping detection")
st.write("⚠️ Drowsiness warning")

st.warning(
    "This is a demo page only. "
    "Live camera detection is not connected yet."
)

st.markdown(
    "[View Python Source Code](https://github.com/Akashmeetha/drowsiness-detection)"
)
