
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Drowsiness Detection", page_icon="🚗")
st.title("🚗 Driver Drowsiness Detection")
st.write("Eye Closure and Yawning Alert Demo")

components.html("""
<div style="font-family:Arial;text-align:center">
  <video id="cam" autoplay playsinline
    style="width:100%;max-width:600px;border-radius:12px"></video>
  <p id="status">Camera not started</p>
  <button id="start">Start Camera</button>
  <p id="alert" style="color:red;font-weight:bold"></p>
  <audio id="alarm" loop>
    <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
  </audio>
  <script>
    const video = document.getElementById('cam');
    const status = document.getElementById('status');
    document.getElementById('start').onclick = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({video:true});
        video.srcObject = stream;
        status.textContent = 'Camera is running';
      } catch (e) {
        status.textContent = 'Camera permission denied';
      }
    };
  </script>
</div>
""", height=500)

st.info("Camera preview only. Eye and yawning detection still need to be implemented.")
