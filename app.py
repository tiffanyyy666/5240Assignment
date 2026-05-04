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
    with st.spinner("🤖 Loading story-writing AI..."):
        story_pipe = pipeline("text-generation", model="gpt2")
    
    # Create a kid-friendly prompt
    prompt = f"Once upon a time, there was a picture of {text}. Write a short and fun story for young children (50 to 100 words): "
    
    # Generate story with length control
    result = story_pipe(
        prompt, 
        max_new_tokens=120,
        do_sample=True, 
        temperature=0.85,
        top_p=0.95
    )
    
    # Extract the generated text
    full_story = result[0]['generated_text']
    
    # Remove the prompt part if it appears in output
    if prompt in full_story:
        story = full_story.replace(prompt, "").strip()
    else:
        story = full_story.strip()
    
    # Ensure story length is between 50-100 words
    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]) + " ... The end!"
    
    # Add a friendly ending if missing
    if not story.endswith((".", "!", "?")):
        story += " The end!"
    
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
    
    # App title and description (using st.title and st.write from class demo)
    st.title("🎨✨ Turn Your Picture into a Magical Story! ✨🎨")
    st.write("Welcome to the **Kids Storyteller App**! Upload a picture and I'll write a story just for you.")
    
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
    
    # File uploader (using st.file_uploader from class demo)
    uploaded_image = st.file_uploader(
        "Upload an image", 
        type=["jpg", "jpeg", "png"],
        help="Choose a picture from your computer - any picture works!"
    )
    
    # Display image with spinner (class demo style)
    if uploaded_image is not None:
        with st.spinner("Loading image..."):
            time.sleep(0.5)  # Simulate a small delay (from class demo)
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        st.divider()
        
        # ===== Stage 1: Image to Text =====
        st.subheader("📝 Step 1: Understanding your picture...")
        with st.spinner("Looking at your picture..."):
            description = img2text(image)  # Pass PIL Image directly
        
        st.success("✅ I see what's in your picture!")
        with st.expander("🔍 Click to see what I see"):
            st.write(f"*{description}*")
        
        # ===== Stage 2: Text to Story =====
        st.subheader("📖 Step 2: Writing your story...")
        with st.spinner("Writing a magical story just for you..."):
            story = text2story(description)
        
        st.success("✅ Your story is ready!")
        st.markdown("**Your 50-100 word story:**")
        st.markdown(f"> {story}")
        
        # Show word count
        word_count = len(story.split())
        st.caption(f"📏 Word count: {word_count} words")
        
        # Warning if length is off
        if word_count < 30:
            st.warning("⚠️ The story is a bit short. Try uploading a different picture!")
        elif word_count > 120:
            st.info("📚 That's a longer story - enjoy!")
        
        # ===== Stage 3: Text to Audio =====
        st.subheader("🔊 Step 3: Listen to your story...")
        with st.spinner("Preparing audio..."):
            audio_data = text2audio(story)
        
        st.success("✅ Audio is ready!")
        
        # Play button (using st.button from class demo)
        if st.button("🔊 Play Story"):
            st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
            st.balloons()  # Fun celebration!
        
        # Reset option
        st.divider()
        if st.button("🔄 Start Over - Upload Another Picture"):
            st.rerun()
    
    else:
        # Friendly prompt when no image is uploaded
        st.info("🎈 **Let's get started!** Use the uploader above to choose a picture.")
        
        # Fun examples
        with st.expander("💡 Tips for best stories"):
            st.markdown("""
            - Try pictures with **animals, people, or nature** 🌳
            - **Bright, clear pictures** work best 📸
            - Take a photo of your **favorite toy** 🧸
            - Draw a picture and upload it! 🎨
            """)

if __name__ == "__main__":
    main()
