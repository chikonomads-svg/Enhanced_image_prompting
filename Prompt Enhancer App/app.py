import os
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Prompt Enhancer",
    page_icon="✨",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        min-height: 150px;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("✨ Prompt Enhancer")
st.markdown("Transform your simple prompts into detailed, comprehensive prompts using AI.")
st.markdown("---")

# Azure OpenAI Configuration from environment variables
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://chikoai.cognitiveservices.azure.com/")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Azure OpenAI client
@st.cache_resource
def get_client():
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

# System prompt for enhancing user prompts
SYSTEM_PROMPT = """You are an expert prompt engineer. Your task is to take a user's simple prompt and transform it into a detailed, comprehensive, and well-structured prompt.

When enhancing the prompt, you should:
1. Add specific context and background information
2. Include clear instructions and expected output format
3. Add relevant constraints and guidelines
4. Specify the tone, style, and audience
5. Break down complex requests into step-by-step instructions
6. Include examples if helpful
7. Add any necessary clarifying questions or assumptions

Make the enhanced prompt detailed enough to get high-quality results but concise enough to be practical."""

# User input section
st.subheader("📝 Enter Your Prompt")
user_prompt = st.text_area(
    "Type your simple prompt here:",
    placeholder="Example: Write a story about a cat",
    help="Enter a basic prompt that you want to enhance and make more detailed."
)

# Enhancement options
st.subheader("⚙️ Options")
col1, col2 = st.columns(2)

with col1:
    prompt_tone = st.selectbox(
        "Desired Tone:",
        ["Professional", "Creative", "Casual", "Academic", "Technical"],
        index=0
    )

with col2:
    detail_level = st.select_slider(
        "Detail Level:",
        options=["Minimal", "Moderate", "Detailed", "Very Detailed"],
        value="Detailed"
    )

# Enhance button
st.markdown("---")
enhance_button = st.button("✨ Enhance Prompt", type="primary", use_container_width=True)

# Process the prompt
if enhance_button:
    if not user_prompt.strip():
        st.error("⚠️ Please enter a prompt to enhance.")
    else:
        with st.spinner("🔄 Enhancing your prompt..."):
            try:
                # Check if API key is configured
                if not subscription_key:
                    st.error("❌ Azure OpenAI API key not found. Please check your .env file.")
                    st.stop()
                
                client = get_client()
                
                # Create the enhancement request
                enhancement_request = f"""Original Prompt: {user_prompt}

Please enhance this prompt with the following specifications:
- Tone: {prompt_tone}
- Detail Level: {detail_level}

Provide only the enhanced prompt without any additional commentary."""

                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": enhancement_request,
                        }
                    ],
                    model=deployment,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                enhanced_prompt = response.choices[0].message.content
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Enhanced Prompt")
                
                # Create a nice container for the result
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown(enhanced_prompt)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Copy button using code block
                st.markdown("#### 📋 Copy Enhanced Prompt")
                st.code(enhanced_prompt, language="markdown")
                
                # Usage stats
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Length", len(user_prompt))
                with col2:
                    st.metric("Enhanced Length", len(enhanced_prompt))
                with col3:
                    expansion = round(len(enhanced_prompt) / len(user_prompt), 1) if len(user_prompt) > 0 else 0
                    st.metric("Expansion Ratio", f"{expansion}x")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please check your Azure OpenAI configuration and try again.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by Azure OpenAI GPT-5-mini</p>",
    unsafe_allow_html=True
)