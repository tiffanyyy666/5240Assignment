# Program title: Storytelling App for Kids (3-10 years old)
# Author: Tiffany Yao
# Description: Upload an image, generate a 50-100 word story, and listen to audio

import streamlit as st
from PIL import Image
import time
from transformers import pipeline

# ------------------ Parameters ------------------
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "roneneldan/TinyStories-33M"
AUDIO_MODEL = "Matthijs/mms-tts-eng"


# ------------------ Functions ------------------

def img2text(image):
    """
    Convert PIL Image to text description using BLIP model
    Uses the image directly in memory (no need to save to disk)
    """
    with st.spinner("🤖 Loading AI model to understand pictures..."):
        # 改动3：修复 pipeline 类型
        image_to_text_model = pipeline("image-to-text", model=CAPTION_MODEL)

    text = image_to_text_model(image)[0]["generated_text"]
    return text


def finish_sentence(text):
    """Trim generated text to the last complete sentence."""
    for mark in [".", "!", "?"]:
        pos = text.rfind(mark)
        if pos > 30:
            return text[: pos + 1].strip()
    return text.strip() + "."


def text2story(scenario):
    """
    Expand the image description into a 50-100 word story suitable for kids aged 3-10
    Uses 3-part generation for more coherent storytelling
    """
    clean_text = scenario.replace("illustration", "").replace("Illustration", "").strip()

    with st.spinner("🤖 Loading story-writing AI..."):
        story_pipe = pipeline("text-generation", model=STORY_MODEL)

    part1_prompt = (
        f"A story for little kids.\n"
        f"Once upon a time, there was {clean_text}. "
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

    part2_raw = story_pipe(
        part1 + " One day, ",
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        return_full_text=False,
    )[0]["generated_text"]
    part2 = finish_sentence(part2_raw)

    part3_raw = story_pipe(
        part2 + " In the end, ",
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        return_full_text=False,
    )[0]["generated_text"]
    part3 = finish_sentence(part3_raw)

    story = f"{part1} {part2} {part3}"

    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]) + " ..."

    return story


def text2audio(story_text):
    """
    Convert story text to audio data using MMS TTS model
    """
    with st.spinner("🤖 Loading audio AI to read your story..."):
        audio_pipe = pipeline("text-to-audio", model=AUDIO_MODEL)

    audio_data = audio_pipe(story_text)
    return audio_data


# ------------------ Main ------------------

def main():
    st.set_page_config(
        page_title="Kids Storyteller 🎈",
        page_icon="📖",
        layout="centered"
    )

    st.header("✨ Turn Your Picture into a Magical Story! ✨")
    st.subheader("Welcome to the **Kids Storyteller App**! Upload a picture and I'll write a story just for you.")

    # Instructions
    st.markdown("""
    ### 📖 How it works:
    1. 📸 Upload or take a picture
    2. 👀 I look at what's in your picture
    3. 📝 I write a **50-100 word story**
    4. 🔊 You can listen to the story!

    *Perfect for kids aged 3-10* 🧸
    """)

    st.divider()

    source = st.radio(
        "Choose how to get your picture:",
        ["📁 Upload an image", "📷 Take a photo"],
        horizontal=True
    )

    uploaded_image = None
    
    if source == "📁 Upload an image":
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            help="Choose a picture from your device"
        )
    else: 
        uploaded_image = st.camera_input("Take a picture with your camera")

    if "story" not in st.session_state:
        st.session_state.story = None
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None
    if "description" not in st.session_state:
        st.session_state.description = None
    if "last_filename" not in st.session_state:
        st.session_state.last_filename = None

    if uploaded_image is not None:

        if source == "📷 Take a photo":
            import time as time_module
            file_name = f"camera_photo_{int(time_module.time())}.jpg"
        else:
            file_name = uploaded_image.name

        if st.session_state.last_filename != file_name:
            st.session_state.story = None
            st.session_state.audio_data = None
            st.session_state.description = None
            st.session_state.last_filename = file_name

        with st.spinner("Loading image..."):
            time.sleep(0.5)
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        st.divider()

        # Image to Text 
        st.subheader("📝 Step 1: Understanding your picture...")

        if st.session_state.description is None:
            with st.spinner("Looking at your picture..."):
                st.session_state.description = img2text(image)

        st.success("✅ I see what's in your picture!")
        with st.expander("🔍 Click to see what I see"):
            st.write(f"*{st.session_state.description}*")

        # Text to Story 
        st.subheader("📖 Step 2: Writing your story...")

        if st.session_state.story is None:
            with st.spinner("Writing a magical story just for you..."):
                st.session_state.story = text2story(st.session_state.description)

        st.success("✅ Your story is ready!")
        st.markdown("**Your 50-100 word story:**")
        st.markdown(f"> {st.session_state.story}")  

        word_count = len(st.session_state.story.split())
        st.caption(f"📏 Word count: {word_count} words")

        if word_count < 50:
            st.warning("⚠️ The story is a bit short. Try uploading a different picture!")
        elif word_count > 100:
            st.info("📚 That's a longer story - enjoy!")

        # Text to Audio 
        st.subheader("🔊 Step 3: Listen to your story...")

        if st.session_state.audio_data is None:
            with st.spinner("Preparing audio..."):
                st.session_state.audio_data = text2audio(st.session_state.story)

        st.success("✅ Audio is ready!")

        if st.button("🔊 Play Story"):
            st.audio(st.session_state.audio_data["audio"], sample_rate=st.session_state.audio_data["sampling_rate"])
            st.balloons()  

    else:
        st.info("🎈 **Let's get started!** Choose an option above to get your picture.")

        with st.expander("🤔 Not sure what to upload?"):
            st.markdown("Try taking a photo of your **favorite toy** 🧸")


if __name__ == "__main__":
    main()
