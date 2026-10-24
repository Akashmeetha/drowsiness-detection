
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗")
st.title("🚗 Driver Drowsiness Detection")
st.write("Eye Closure and Yawning Detection with Sound Alert")

components.html("""
<div style="font-family:Arial;text-align:center">
  <video id="cam" autoplay playsinline
    style="width:100%;max-width:600px;border-radius:12px"></video>
  <h3 id="status">Click Start Detection</h3>
  <button id="start" style="padding:12px 20px">Start Detection</button>
  <button id="stop" style="padding:12px 20px">Stop</button>
  <p id="alert" style="color:red;font-size:22px;font-weight:bold"></p>
  <audio id="alarm" loop>
    <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
  </audio>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>

<script>
const video = document.getElementById('cam');
const statusText = document.getElementById('status');
const alertText = document.getElementById('alert');
const alarm = document.getElementById('alarm');
let camera = null, faceMesh = null;
let closedFrames = 0, mouthFrames = 0, running = false;

function distance(a,b) {
  return Math.hypot(a.x-b.x, a.y-b.y);
}

async function startDetection() {
  try {
    alarm.pause();
    alarm.currentTime = 0;
    await alarm.play();
    alarm.pause();

    faceMesh = new FaceMesh({
      locateFile: file =>
        'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/' + file
    });

    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    faceMesh.onResults(results => {
      if (!running) return;

      if (!results.multiFaceLandmarks ||
          results.multiFaceLandmarks.length === 0) {
        statusText.textContent = 'Face not detected';
        alertText.textContent = '';
        alarm.pause();
        return;
      }

      const p = results.multiFaceLandmarks[0];

      // Eye opening ratio using eye landmarks
      const leftEye = distance(p[159],p[145]) /
                      (distance(p[33],p[133]) || 1);
      const rightEye = distance(p[386],p[374]) /
                       (distance(p[362],p[263]) || 1);
      const eyeRatio = (leftEye + rightEye) / 2;

      // Mouth opening ratio
      const mouthRatio = distance(p[13],p[14]) /
                         (distance(p[78],p[308]) || 1);

      if (eyeRatio < 0.18) closedFrames++;
      else closedFrames = 0;

      if (mouthRatio > 0.35) mouthFrames++;
      else mouthFrames = 0;

      let message = 'Status: Awake';
      if (closedFrames > 15) message = '⚠️ Drowsiness: Eyes closed';
      if (mouthFrames > 15) message = '⚠️ Possible yawning detected';

      statusText.textContent = message;

      if (closedFrames > 15 || mouthFrames > 15) {
        alertText.textContent = '⚠️ WAKE UP! Please stay alert!';
        alarm.play().catch(() => {});
      } else {
        alertText.textContent = '';
        alarm.pause();
        alarm.currentTime = 0;
      }
    });

    const stream = await navigator.mediaDevices.getUserMedia({
      video: true
    });
    video.srcObject = stream;
    await video.play();

    running = true;
    camera = new Camera(video, {
      onFrame: async () => {
        if (running) await faceMesh.send({image: video});
      },
      width: 640,
      height: 480
    });
    camera.start();
    statusText.textContent = 'Detection running';
  } catch (e) {
    statusText.textContent = 'Could not start. Allow camera and try again.';
    console.error(e);
  }
}

document.getElementById('start').onclick = startDetection;

document.getElementById('stop').onclick = () => {
  running = false;
  if (camera) camera.stop();
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }
  alarm.pause();
  alarm.currentTime = 0;
  statusText.textContent = 'Detection stopped';
  alertText.textContent = '';
};
</script>
""", height=650)

st.warning("Demo only. Detection thresholds need testing and this is not a certified driver-safety system.")
