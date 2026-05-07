# Program title: Storytelling App
# Import part
import streamlit as st
from PIL import Image
from transformers import pipeline

# ------------------ Parameters ------------------
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "roneneldan/TinyStories-33M"
AUDIO_MODEL = "Matthijs/mms-tts-eng"


# ------------------ Functions ------------------
def img2text(image_path):
    image_to_text_model = pipeline("image-text-to-text", model=CAPTION_MODEL)
    image = Image.open(image_path)
    text = image_to_text_model(image, text="a picture of")[0]["generated_text"]
    # Remove the prompt prefix from the caption
    prefix = "a picture of"
    if text.lower().startswith(prefix):
        text = text[len(prefix):].strip(", ")
    return text

def text2story(scenario):
    story_pipe = pipeline("text-generation", model=STORY_MODEL)

    # 1
    part1_prompt = (
        f"A story for little kids.\n"
        f"Once upon a time, there was {scenario}. "
    )
    part1_raw = story_pipe(
        part1_prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        return_full_text=False,
    )[0]["generated_text"]
    part1 = finish_sentence(part1_raw)

    # 2
    part2_raw = story_pipe(
        part1 + " One day, ",
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        return_full_text=False,
    )[0]["generated_text"]
    part2 = finish_sentence(part2_raw)

    # 3
    part3_raw = story_pipe(
        part2 + " In the end, ",
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        return_full_text=False,
    )[0]["generated_text"]
    part3 = finish_sentence(part3_raw)

    return f"{part1} {part2} {part3}"

def text2audio(story_text):
    audio_pipe = pipeline("text-to-audio", model=AUDIO_MODEL)
    return audio_pipe(story_text)


def finish_sentence(text):
    """Trim generated text to the last complete sentence."""
    for mark in [".", "!", "?"]:
        pos = text.rfind(mark)
        if pos > 30:
            return text[: pos + 1].strip()
    return text.strip() + "."


# ------------------ Main ------------------
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🤖")
st.header("ISOM5240: Turn Your Image to Audio Story")

source = st.radio("Choose an image source", ["Upload an image 📁", "Take a photo 📷"], horizontal=True)

uploaded_file = None
if source == "Upload an image 📁":
    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a picture with your camera")

# Initialize session state
for key in ["scenario", "story", "audio_data", "last_file_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Uploaded Image 🖼️", use_column_width=True)

    # Only re-run pipeline when the image changes
    file_changed = st.session_state.last_file_name != uploaded_file.name

    # Stage 1: Image to Text
    if file_changed or st.session_state.scenario is None:
        st.text("Processing img2text...")
        st.session_state.scenario = img2text(uploaded_file.name)

    st.write(f"**Scenario:** {st.session_state.scenario}")

    # Stage 2: Text to Story
    if file_changed or st.session_state.story is None:
        st.text("Generating a story...")
        st.session_state.story = text2story(st.session_state.scenario)

    st.write(f"**Story:** {st.session_state.story}")

    # Stage 3: Story to Audio
    if file_changed or st.session_state.audio_data is None:
        st.text("Generating audio data...")
        st.session_state.audio_data = text2audio(st.session_state.story)

    st.session_state.last_file_name = uploaded_file.name

    # Show audio player persistently (not inside a button callback)
    audio_array = st.session_state.audio_data["audio"]
    sample_rate = st.session_state.audio_data["sampling_rate"]
    st.audio(audio_array, sample_rate=sample_rate)
