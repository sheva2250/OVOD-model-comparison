import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import time

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
        # Grounding DINO hasilnya sudah punya label dari processor
        return results[0]
    else:
        results = processor.post_process_object_detection(
            outputs,
            threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        # OWLv2: Perlu ambil label dari inputs.input_ids dan decode
        labels = [label.strip() for label in prompt.split('.') if label.strip()]
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        pred_labels = [labels[i] for i in results[0]["labels"]]

        return {"boxes": boxes, "labels": pred_labels, "scores": scores}


# === Visualisasi ===
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

# === UI Streamlit ===
st.title("Zero-Shot Object Detection")

model_option = st.selectbox("Pilih Model", ["Grounding DINO", "OWL-V2"])
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

st.divider()


if model_option == "Grounding DINO":
    prompt = st.text_input("Masukkan Prompt (a dog. a person. blue shirt)", placeholder="Akhiri dengan tanda titik pada setiap prompt.")
else:
    prompt = st.text_input("Masukkan Prompt (a dog, a person, blue shirt)", placeholder="Pisahkan dengan koma untuk setiap prompt")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    st.divider()

    with st.spinner("Mendeteksi objek...", show_time=True):
        result = detect_objects(image, prompt, model_option)
        boxes = result['boxes'].tolist()
        labels = result['labels']
        scores = result['scores'].tolist()
        st.success("Deteksi selesai!")

        col1, col2 = st.columns(2)

        with col1:
            st.header("Hasil Deteksi")
            img_buf = draw_boxes(image, boxes, labels, scores)
            st.image(img_buf, caption="Hasil Deteksi", use_container_width=True)

        with col2:
            st.header("Label Terdeteksi")
            for label, score in zip(labels, scores):
                st.markdown(f"- **{label}** ({score:.2f})")
