import streamlit as st
import cv2
import tempfile
import os
import time
import pandas as pd
import altair as alt
from utils import SmartClassroomModel, get_occupancy_level, get_ac_state

st.set_page_config(page_title="Smart Classroom Edge AI", layout="wide")

# Hide the "Deploy" button
st.markdown("<style>.stDeployButton {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("Smart Classroom Edge AI System")
st.markdown("Object Detection · Occupancy Counting · Automated AC Control")

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Object Detection model...")
def load_model():
    return SmartClassroomModel("model.onnx", "labels.txt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ── Sidebar Configuration ──────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Playback Settings")
playback_speed = st.sidebar.slider("Playback Speed Multiplier", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
st.sidebar.markdown("---")

st.sidebar.subheader("Occupancy Rules")
st.sidebar.markdown("""
| Students | Level |
|---|---|
| 0 (janitors only) | 🟢 LOW – AC OFF |
| < 3 | 🟢 LOW – AC OFF |
| 3 – 9 | 🟡 MEDIUM – AC ON |
| > 9 | 🔴 HIGH – AC ON |
""")
st.sidebar.markdown("---")

st.sidebar.subheader("Detection Sensitivity")
st.sidebar.markdown("Increase to remove phantom detections. Decrease if real people are being missed.")
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", min_value=0.10, max_value=0.95, value=0.50, step=0.05)
st.sidebar.markdown("---")
medium_temp_setting = st.sidebar.number_input("Medium Occupancy Temp (°C)", min_value=16, max_value=30, value=24)
high_temp_setting = st.sidebar.number_input("High Occupancy Temp (°C)", min_value=16, max_value=30, value=20)
st.sidebar.markdown("---")

# ── File Upload ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Classroom Video", type=["mp4", "avi", "mov", "mpeg4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.markdown("### Live Dashboard")

    col1, col2 = st.columns([2, 1])
    with col1:
        video_placeholder = st.empty()
    with col2:
        occupancy_placeholder = st.empty()
        count_placeholder = st.empty()
        ac_state_placeholder = st.empty()
        temperature_placeholder = st.empty()
        runtime_placeholder = st.empty()
        st.markdown("<br>", unsafe_allow_html=True)
        ac_animation_placeholder = st.empty()

    st.markdown("### Occupancy Timeline")
    chart_placeholder = st.empty()

    # ── Session State ──────────────────────────────────────────────────────────
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "current_frame" not in st.session_state:
        st.session_state.current_frame = 0
    if "total_ac_time" not in st.session_state:
        st.session_state.total_ac_time = 0
    if "timeline_data" not in st.session_state:
        st.session_state.timeline_data = []

    colA, colB = st.columns([1, 4])
    with colA:
        if st.button("Start Processing"):
            st.session_state.is_processing = True
            st.session_state.current_frame = 0
            st.session_state.total_ac_time = 0
            st.session_state.timeline_data = []
    with colB:
        if st.button("Stop"):
            st.session_state.is_processing = False

    # ── AC Animation HTML ──────────────────────────────────────────────────────
    ac_html_on = """
<style>
.ac-unit { width: 280px; height: 70px; background: #e0e0e0; border-radius: 8px; border: 2px solid #ccc; position: relative; margin: 10px auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.ac-vent { width: 220px; height: 8px; background: #00d2ff; position: absolute; bottom: 8px; left: 30px; border-radius: 4px; box-shadow: 0 0 10px #00d2ff; }
.wind-line { position: absolute; width: 4px; height: 30px; background: linear-gradient(to bottom, #00d2ff, transparent); bottom: -40px; border-radius: 2px; }
@keyframes blow { 0% { transform: translateY(0); opacity: 0.8; } 100% { transform: translateY(40px); opacity: 0; } }
.w1 { left: 50px; animation: blow 1s infinite; }
.w2 { left: 95px; animation: blow 1s infinite 0.2s; }
.w3 { left: 140px; animation: blow 1s infinite 0.4s; }
.w4 { left: 185px; animation: blow 1s infinite 0.1s; }
.w5 { left: 230px; animation: blow 1s infinite 0.3s; }
</style>
<div style="text-align:center;">
<div class="ac-unit">
<div style="position:absolute; top:10px; left:15px; font-weight:bold; color:#333; font-size:14px;">AC Simulation</div>
<div style="position:absolute; top:10px; right:15px; color:#00d2ff; font-weight:bold; font-size:12px;">ON</div>
<div class="ac-vent"></div>
<div class="wind-line w1"></div>
<div class="wind-line w2"></div>
<div class="wind-line w3"></div>
<div class="wind-line w4"></div>
<div class="wind-line w5"></div>
</div>
</div>
"""

    ac_html_off = """
<style>
.ac-unit-off { width: 280px; height: 70px; background: #e0e0e0; border-radius: 8px; border: 2px solid #ccc; position: relative; margin: 10px auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1); opacity: 0.6; }
.ac-vent-off { width: 220px; height: 8px; background: #555; position: absolute; bottom: 8px; left: 30px; border-radius: 4px; }
</style>
<div style="text-align:center;">
<div class="ac-unit-off">
<div style="position:absolute; top:10px; left:15px; font-weight:bold; color:#333; font-size:14px;">AC Simulation</div>
<div style="position:absolute; top:10px; right:15px; color:#555; font-weight:bold; font-size:12px;">OFF</div>
<div class="ac-vent-off"></div>
</div>
</div>
"""

    # ── Processing Loop ────────────────────────────────────────────────────────
    if st.session_state.is_processing:
        input_source = "Video File"
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps:
            fps = 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, int(playback_speed))

        last_ai_run_frame = -fps  # Force AI run on first frame

        # Latest detection state (persisted between AI runs)
        last_detections = []
        last_student_count = 0
        last_janitor_count = 0
        last_level = 'low'
        last_janitor_only = False

        prev_time = time.time()

        while cap.isOpened() and st.session_state.current_frame < total_frames and st.session_state.is_processing:
            if frame_step > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            # ── 1. Draw bounding boxes on a copy of the frame ─────────────────
            display_frame = frame.copy()
            h_f, w_f = display_frame.shape[:2]

            for det in last_detections:
                bb = det['boundingBox']
                x1 = int(bb['left'] * w_f)
                y1 = int(bb['top'] * h_f)
                x2 = int((bb['left'] + bb['width']) * w_f)
                y2 = int((bb['top'] + bb['height']) * h_f)
                label = det['tagName'].lower()
                color = (0, 200, 0) if label == 'student' else (0, 165, 255)  # green=student, orange=janitor
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{det['tagName']} {det['probability']:.2f}",
                            (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Resize for UI performance
            max_w = 640
            if w_f > max_w:
                scale = max_w / w_f
                display_frame = cv2.resize(display_frame, (max_w, int(h_f * scale)))

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # ── 2. AI Inference (once per video-second) ───────────────────────
            if (st.session_state.current_frame - last_ai_run_frame) >= fps:
                seconds_passed = (st.session_state.current_frame - last_ai_run_frame) / fps if last_ai_run_frame >= 0 else 0

                last_detections, last_student_count, last_janitor_count = model.predict(frame, prob_threshold=confidence_threshold)
                last_level, last_janitor_only = get_occupancy_level(last_student_count, last_janitor_count)
                ac_state, temperature = get_ac_state(last_level, last_janitor_only, medium_temp_setting, high_temp_setting)

                if ac_state == "ON" and last_ai_run_frame >= 0:
                    st.session_state.total_ac_time += seconds_passed

                # Build count summary string
                parts = []
                if last_student_count > 0:
                    parts.append(f"👨‍🎓 {last_student_count} student{'s' if last_student_count != 1 else ''}")
                if last_janitor_count > 0:
                    parts.append(f"🧹 {last_janitor_count} janitor{'s' if last_janitor_count != 1 else ''}")
                count_str = " · ".join(parts) if parts else "No people detected"

                # Timeline
                level_map = {"low": 1, "medium": 2, "high": 3}
                st.session_state.timeline_data.append({
                    "Time (s)": st.session_state.current_frame // int(fps),
                    "Occupancy": level_map.get(last_level, 0),
                    "Level": last_level.upper()
                })

                m, s = divmod(int(st.session_state.total_ac_time), 60)
                h, m = divmod(m, 60)
                runtime_str = f"{h:02d}:{m:02d}:{s:02d}"

                # Janitor-only notice
                if last_janitor_only:
                    occupancy_placeholder.info(f"**Occupancy Level:** LOW 🧹 (Janitor-only — AC forced OFF)")
                else:
                    occupancy_placeholder.info(f"**Occupancy Level:** {last_level.upper()}")

                count_placeholder.markdown(f"**Detected:** {count_str}")
                runtime_placeholder.metric("Total AC Running Time", runtime_str)

                if ac_state == "ON":
                    ac_state_placeholder.success(f"**AC State:** {ac_state}")
                    temperature_placeholder.warning(f"**Temperature:** {temperature}")
                    ac_animation_placeholder.markdown(ac_html_on, unsafe_allow_html=True)
                else:
                    ac_state_placeholder.error(f"**AC State:** {ac_state}")
                    temperature_placeholder.info(f"**Temperature:** {temperature}")
                    ac_animation_placeholder.markdown(ac_html_off, unsafe_allow_html=True)

                if len(st.session_state.timeline_data) > 0:
                    df = pd.DataFrame(st.session_state.timeline_data)
                    chart = alt.Chart(df).mark_line(point=True).encode(
                        x=alt.X('Time (s):Q', title='Time (Seconds)'),
                        y=alt.Y('Occupancy:Q', scale=alt.Scale(domain=[0, 4]),
                                title='Occupancy (1=Low, 2=Medium, 3=High)'),
                        tooltip=['Time (s)', 'Level']
                    ).properties(height=200)
                    chart_placeholder.altair_chart(chart, use_container_width=True)

                last_ai_run_frame = st.session_state.current_frame

            # ── 3. Frame timing ───────────────────────────────────────────────
            st.session_state.current_frame += frame_step
            target_loop_time = (1.0 / fps) * (frame_step / playback_speed)
            elapsed = time.time() - prev_time
            if elapsed < target_loop_time:
                time.sleep(target_loop_time - elapsed)
            prev_time = time.time()

        cap.release()

        if st.session_state.current_frame >= total_frames:
            st.session_state.is_processing = False
            occupancy_placeholder.info("**Occupancy Level:** LOW (Video Ended)")
            ac_state_placeholder.error("**AC State:** OFF")
            temperature_placeholder.info("**Temperature:** —")
            ac_animation_placeholder.markdown(ac_html_off, unsafe_allow_html=True)
            st.success("Video playback completed. AC has been turned off.")

        try:
            os.remove(video_path)
        except Exception:
            pass
