import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import time
import base64

# === Inisialisasi session state ===
if 'history' not in st.session_state:
    st.session_state['history'] = []

# === Model Loader ===
@st.cache_resource
def load_grounding_dino():
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

@st.cache_resource
def load_owlv2():
    model_id = "google/owlv2-base-patch16-ensemble"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

# === Deteksi Berdasarkan Model ===
def detect_objects(image: Image.Image, prompt: str, model_option: str):
    if model_option == "Grounding DINO":
        processor, model, device = load_grounding_dino()
    else:
        processor, model, device = load_owlv2()

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    if model_option == "Grounding DINO":
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        return results[0]
    else:
        results = processor.post_process_object_detection(
            outputs,
            threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        labels = [label.strip() for label in prompt.split('.') if label.strip()]
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        pred_labels = [labels[i] for i in results[0]["labels"]]

        return {"boxes": boxes, "labels": pred_labels, "scores": scores}

# === Visualisasi dan Buffering ===
def draw_boxes(image: Image.Image, boxes, labels, scores):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{label} ({score:.2f})", fontsize=10, color='white', bbox=dict(facecolor='red', alpha=0.5))
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

def image_to_base64(buf):
    return base64.b64encode(buf.getvalue()).decode()

# === UI Streamlit ===
st.markdown("Nama: Sheva Aqila Ramadhan - 2304130083")
st.markdown("Dikerjakan oleh saya sendiri dan 10% AI")

st.divider()

st.title("Zero-Shot Object Detection")

model_option = st.selectbox("Pilih Model", ["Grounding DINO", "OWL-V2"])

if model_option == "Grounding DINO":
    prompt = st.text_input("Masukkan Prompt (a dog. a person. blue shirt)", placeholder="Akhiri dengan tanda titik pada setiap prompt.")
else:
    prompt = st.text_input("Masukkan Prompt (a dog, a person, blue shirt)", placeholder="Pisahkan dengan koma untuk setiap prompt")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

st.divider()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    st.divider()

    if prompt.strip():
        if st.button("Deteksi Objek"):
            with st.spinner("Mendeteksi objek...", show_time=True):
                start_time = time.time()
                result = detect_objects(image, prompt, model_option)
                end_time = time.time()
                duration = end_time - start_time

                boxes = result['boxes'].tolist()
                labels = result['labels']
                scores = result['scores'].tolist()

                st.success(f"Deteksi selesai dalam {duration:.2f} detik!")

                img_buf = draw_boxes(image, boxes, labels, scores)
                st.image(img_buf, caption="Hasil Deteksi", use_container_width=True)

                st.header("Label Terdeteksi")
                for label, score in zip(labels, scores):
                    st.markdown(f"- **{label}** ({score:.2f})")

                st.text(f"Prompt: {prompt}")

                # Simpan ke history (maksimal 10 item)
                st.session_state['history'].insert(0, {
                    "prompt": prompt,
                    "model": model_option,
                    "duration": duration,
                    "labels": list(zip(labels, scores)),
                    "image_b64": image_to_base64(img_buf)
                })
                st.session_state['history'] = st.session_state['history'][:10]
    else:
        st.warning("Masukkan prompt terlebih dahulu sebelum mendeteksi.")

# === Sidebar History ===
with st.sidebar:
    st.header("Riwayat Prediksi")
    if st.session_state['history']:
        for i, entry in enumerate(st.session_state['history']):
            with st.expander(f"Prediksi #{i+1} ({entry['model']})"):
                st.markdown(f"**Prompt**: {entry['prompt']}")
                st.markdown(f"**Durasi**: {entry['duration']:.2f} detik")
                for lbl, sc in entry['labels']:
                    st.markdown(f"- {lbl} ({sc:.2f})")
                st.image(f"data:image/png;base64,{entry['image_b64']}", use_container_width=True)
    else:
        st.info("Belum ada prediksi yang dilakukan.")
