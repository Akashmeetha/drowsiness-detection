
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Drowsiness Detection", page_icon="🚗")

st.title("🚗 Driver Drowsiness Detection")
st.write("Live Camera Demo")

st.warning("Allow camera access when your browser asks.")

components.html("""
<div style="font-family:Arial;text-align:center">
  <video id="cam" autoplay playsinline
    style="width:100%;max-width:600px;border-radius:12px"></video>
  <p id="status">Camera not started</p>
  <button id="start">Start Camera</button>
  <script>
    const video = document.getElementById('cam');
    const status = document.getElementById('status');
    document.getElementById('start').onclick = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({video:true});
        video.srcObject = stream;
        status.textContent = 'Camera is running';
      } catch (e) {
        status.textContent = 'Camera permission denied or unavailable';
      }
    };
  </script>
</div>
""", height=500)
