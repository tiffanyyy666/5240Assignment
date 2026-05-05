# Program title: Storytelling App for Kids (3-10 years old)
# Author: Tiffany Yao
# Description: Upload an image, generate a 50-100 word story, and listen to audio

import streamlit as st
from PIL import Image
import time
from transformers import pipeline

# Function Part

def img2text(image):
    """
    Convert PIL Image to text description using BLIP model
    Uses the image directly in memory (no need to save to disk)
    """
    with st.spinner("🤖 Loading AI model to understand pictures..."):
        image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Pass the PIL Image directly to the pipeline
    text = image_to_text_model(image)[0]["generated_text"]
    return text

def text2story(text):
    """
    Expand the image description into a 50-100 word story suitable for kids aged 3-10
    """
    # 确保输入是一个完整的句子
    if not text.endswith((".", "!", "?")):
        text = text + "."
    
    with st.spinner("🤖 Loading story-writing AI..."):
        # 用完整版 gpt2，不是 distilgpt2
        story_pipe = pipeline("text-generation", model="gpt2")
    
    # 清晰的故事开头
    prompt = f"Once upon a time, {text} "
    
    result = story_pipe(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2  # 减少重复
    )
    
    full_story = result[0]['generated_text']
    
    # 去掉 prompt 部分
    if full_story.startswith(prompt):
        story = full_story[len(prompt):].strip()
    else:
        story = full_story.strip()
    
    # 如果还是太长，截断
    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]) + " ..."
    
    # 确保结尾标点
    if not story.endswith((".", "!", "?")):
        story += "."
    
    return story

def text2audio(story_text):
    """
    Convert story text to audio data using MMS TTS model
    """
    with st.spinner("🤖 Loading audio AI to read your story..."):
        audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    
    audio_data = audio_pipe(story_text)
    return audio_data

# Main Part

def main():
    # Page configuration
    st.set_page_config(
        page_title="Kids Storyteller 🎈", 
        page_icon="📖",
        layout="centered"
    )
    
    # App title and description
    st.header("✨ Turn Your Picture into a Magical Story! ✨")
    st.subheader("Welcome to the **Kids Storyteller App**! Upload a picture and I'll write a story just for you.")
    
    # Instructions
    st.markdown("""
    ### 📖 How it works:
    1. 📸 Upload a picture
    2. 👀 I look at what's in your picture
    3. 📝 I write a **50-100 word story**
    4. 🔊 You can listen to the story!
    
    *Perfect for kids aged 3-10* 🧸
    """)
    
    st.divider()

    uploaded_image = st.file_uploader(
        "Upload an image", 
        type=["jpg", "jpeg", "png"],
        help="Choose a picture from your device"
    )
    
    if "story" not in st.session_state:
        st.session_state.story = None
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None
    if "description" not in st.session_state:
        st.session_state.description = None
    if "last_filename" not in st.session_state:
        st.session_state.last_filename = None
    
    # Display image with spinner
    if uploaded_image is not None:
        
        if st.session_state.last_filename != uploaded_image.name:
            st.session_state.story = None
            st.session_state.audio_data = None
            st.session_state.description = None
            st.session_state.last_filename = uploaded_image.name
        
        with st.spinner("Loading image..."):
            time.sleep(0.5) 
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        st.divider()
        
        # ===== Stage 1: Image to Text =====
        st.subheader("📝 Step 1: Understanding your picture...")
        
        if st.session_state.description is None:
            with st.spinner("Looking at your picture..."):
                st.session_state.description = img2text(image)
        
        st.success("✅ I see what's in your picture!")
        with st.expander("🔍 Click to see what I see"):
            st.write(f"*{st.session_state.description}*")
        
        # ===== Stage 2: Text to Story =====
        st.subheader("📖 Step 2: Writing your story...")
        
        if st.session_state.story is None:
            with st.spinner("Writing a magical story just for you..."):
                st.session_state.story = text2story(st.session_state.description)
        
        st.success("✅ Your story is ready!")
        st.markdown("**Your 50-100 word story:**")
        st.markdown(f"> {st.session_state.story}")
        
        # Show word count
        word_count = len(st.session_state.story.split())
        st.caption(f"📏 Word count: {word_count} words")
        
        # Warning if length is off
        if word_count < 50:
            st.warning("⚠️ The story is a bit short. Try uploading a different picture!")
        elif word_count > 100:
            st.info("📚 That's a longer story - enjoy!")
        
        # ===== Stage 3: Text to Audio =====
        st.subheader("🔊 Step 3: Listen to your story...")
        
        if st.session_state.audio_data is None:
            with st.spinner("Preparing audio..."):
                st.session_state.audio_data = text2audio(st.session_state.story)
        
        st.success("✅ Audio is ready!")
        
        # Play button (using st.button from class demo)
        if st.button("🔊 Play Story"):
            st.audio(st.session_state.audio_data["audio"], sample_rate=st.session_state.audio_data["sampling_rate"])
            st.balloons()  # Fun celebration!
    
    else:
        # Friendly prompt when no image is uploaded
        st.info("🎈 **Let's get started!** Use the uploader above to choose a picture.")
        
        # Fun examples
        with st.expander("🤔 Not sure what to upload?"):
            st.markdown("Try taking a photo of your **favorite toy** 🧸")

if __name__ == "__main__":
    main()
