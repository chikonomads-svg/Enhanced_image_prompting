"""
Deep Research App - Streamlit UI
User-friendly interface for deep web research using Tavily API
"""

import streamlit as st
import sys
import os

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_logic import (
    DeepResearcher, 
    ResearchResult, 
    format_research_output,
    TavilyResearchError
)

# Page configuration
st.set_page_config(
    page_title="Deep Research App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #1E88E5 !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
    }
    .sub-header {
        font-size: 1.2rem !important;
        color: #666 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
    }
    .research-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
    }
    .key-point {
        background-color: #e3f2fd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #2196F3;
    }
    .source-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        font-size: 0.9rem;
    }
    .error-box {
        background-color: #ffebee;
        border-radius: 5px;
        padding: 15px;
        border-left: 4px solid #f44336;
        color: #c62828;
    }
    .success-box {
        background-color: #e8f5e9;
        border-radius: 5px;
        padding: 15px;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 10px 30px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'research_history' not in st.session_state:
        st.session_state.research_history = []
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'is_researching' not in st.session_state:
        st.session_state.is_researching = False


def display_header():
    """Display the app header"""
    st.markdown('<p class="main-header">🔍 Deep Research App</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Tavily AI - Deep Web Research & Analysis</p>', unsafe_allow_html=True)


def display_sidebar():
    """Display the sidebar with settings and history"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Research depth selection
        search_depth = st.selectbox(
            "Research Depth",
            options=["advanced", "basic"],
            index=0,
            key="search_depth_select",
            help="Advanced provides deeper research with more sources"
        )
        
        # Max results slider
        max_results = st.slider(
            "Max Results per Query",
            min_value=5,
            max_value=20,
            value=10,
            key="max_results_slider",
            help="Number of results to fetch per search query"
        )
        
        st.divider()
        
        # Research History
        st.header("📜 Research History")
        if st.session_state.research_history:
            for i, item in enumerate(reversed(st.session_state.research_history[-10:]), 1):
                with st.expander(f"{i}. {item['topic'][:40]}..."):
                    st.write(f"**Time:** {item['timestamp']}")
                    st.write(f"**Key Points:** {item['key_points_count']}")
                    st.write(f"**Sources:** {item['sources_count']}")
                    if st.button(f"Load This Research", key=f"load_{i}"):
                        st.session_state.current_result = item['result']
                        st.rerun()
        else:
            st.info("No research history yet. Start researching!")
        
        st.divider()
        
        # Clear history button
        if st.session_state.research_history and st.button("🗑️ Clear History"):
            st.session_state.research_history = []
            st.rerun()
        
        # About section
        st.header("ℹ️ About")
        st.markdown("""
        **Deep Research App** uses Tavily's AI-powered search to:
        - 🔍 Perform comprehensive web research
        - 📊 Extract key insights and points
        - 📚 Cite reliable sources
        - ⚡ Deliver results quickly
        
        Perfect for researching AI tools, new technologies, and more!
        """)


def validate_input(topic: str) -> tuple[bool, str]:
    """
    Validate user input
    
    Args:
        topic: The research topic
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not topic or not topic.strip():
        return False, "Please enter a research topic."
    
    topic = topic.strip()
    
    if len(topic) < 3:
        return False, "Topic is too short. Please enter at least 3 characters."
    
    if len(topic) > 500:
        return False, "Topic is too long. Please limit to 500 characters."
    
    # Check for valid characters (alphanumeric, spaces, common punctuation)
    if not re.match(r'^[\w\s\-\'".,!?()[\]{}:;@#$%&*+=/<>]+$', topic):
        return False, "Topic contains invalid characters."
    
    return True, ""


def perform_research(topic: str, search_depth: str, max_results: int) -> ResearchResult:
    """
    Perform research with error handling
    
    Args:
        topic: Research topic
        search_depth: Search depth level
        max_results: Maximum results to fetch
        
    Returns:
        ResearchResult object
    """
    try:
        researcher = DeepResearcher()
        result = researcher.research(
            topic=topic,
            search_depth=search_depth,
            max_results=max_results
        )
        return result
    
    except TavilyResearchError as e:
        error_result = ResearchResult(topic=topic)
        error_result.error_message = str(e)
        return error_result
    
    except Exception as e:
        error_result = ResearchResult(topic=topic)
        error_result.error_message = f"An unexpected error occurred: {str(e)}"
        return error_result


def display_research_results(result: ResearchResult):
    """
    Display research results in a formatted way
    
    Args:
        result: ResearchResult object
    """
    if not result.success:
        st.markdown(f'<div class="error-box">❌ {result.error_message}</div>', unsafe_allow_html=True)
        return
    
    # Success message
    st.markdown(f'<div class="success-box">✅ Research completed successfully!</div>', unsafe_allow_html=True)
    
    # Summary section
    st.subheader("📋 Summary")
    with st.container():
        st.markdown('<div class="research-card">', unsafe_allow_html=True)
        st.write(result.summary)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Key Points section
    st.subheader(f"🎯 Key Points ({len(result.key_points)} found)")
    with st.container():
        for i, point in enumerate(result.key_points, 1):
            st.markdown(f'<div class="key-point">{i}. {point}</div>', unsafe_allow_html=True)
    
    # Sources section
    st.subheader(f"📚 Sources ({len(result.sources)} found)")
    with st.container():
        for i, source in enumerate(result.sources, 1):
            st.markdown(f'''
            <div class="source-card">
                <strong>{i}. {source['title']}</strong><br>
                <a href="{source['url']}" target="_blank">{source['url']}</a>
            </div>
            ''', unsafe_allow_html=True)
    
    # Search Details
    with st.expander("🔧 Search Details"):
        st.write(f"**Topic:** {result.topic}")
        st.write(f"**Search Queries Used:** {len(result.search_queries)}")
        st.write(f"**Total Sources:** {len(result.sources)}")
        st.write(f"**Key Points Extracted:** {len(result.key_points)}")
        st.write(f"**Research Time:** {result.timestamp}")
        
        st.write("**Queries Executed:**")
        for i, query in enumerate(result.search_queries, 1):
            st.write(f"  {i}. {query}")
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        # Download as text
        text_output = format_research_output(result)
        st.download_button(
            label="📥 Download as Text",
            data=text_output,
            file_name=f"research_{result.topic.replace(' ', '_')[:30]}.txt",
            mime="text/plain"
        )
    
    with col2:
        # Create markdown version
        md_output = f"# Research Report: {result.topic}\n\n"
        md_output += f"## Summary\n\n{result.summary}\n\n"
        md_output += "## Key Points\n\n"
        for i, point in enumerate(result.key_points, 1):
            md_output += f"{i}. {point}\n\n"
        md_output += "## Sources\n\n"
        for source in result.sources:
            md_output += f"- [{source['title']}]({source['url']})\n"
        
        st.download_button(
            label="📥 Download as Markdown",
            data=md_output,
            file_name=f"research_{result.topic.replace(' ', '_')[:30]}.md",
            mime="text/markdown"
        )


def display_example_topics():
    """Display example topics for inspiration"""
    with st.expander("💡 Need inspiration? Click for example topics"):
        examples = [
            "OpenAI GPT-5 latest features",
            "Claude AI Anthropic capabilities",
            "LangChain framework tutorial",
            "AutoGPT autonomous agents",
            "Midjourney v6 image generation",
            "Hugging Face transformers",
            "Pinecone vector database",
            "CrewAI multi-agent systems",
            "Ollama local LLM deployment",
            "Stable Diffusion XL model"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"🔍 {example}", key=f"example_{i}", use_container_width=True):
                    st.session_state.example_topic = example
                    st.rerun()


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    st.divider()
    
    # Research input section
    st.subheader("📝 Enter Your Research Topic")
    
    # Check if an example was clicked
    default_value = ""
    if 'example_topic' in st.session_state:
        default_value = st.session_state.example_topic
        del st.session_state.example_topic
    
    topic = st.text_input(
        "What would you like to research?",
        value=default_value,
        placeholder="e.g., OpenAI GPT-5, LangChain framework, Claude AI...",
        help="Enter any AI tool, technology, or topic you want to research"
    )
    
    # Example topics
    display_example_topics()
    
    # Get settings from sidebar
    with st.sidebar:
        pass  # Settings are already displayed in sidebar
    
    # Re-get settings for use
    sidebar_settings = st.sidebar
    search_depth = st.session_state.get('search_depth', 'advanced')
    max_results = st.session_state.get('max_results', 10)
    
    # Actually get the current values from sidebar widgets
    # We need to retrieve them from the sidebar context
    with st.sidebar:
        # These are already rendered, but we need their values
        pass
    
    # Research button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        research_button = st.button("🔍 Start Research", use_container_width=True)
    
    # Perform research
    if research_button:
        # Validate input
        is_valid, error_msg = validate_input(topic)
        
        if not is_valid:
            st.markdown(f'<div class="error-box">⚠️ {error_msg}</div>', unsafe_allow_html=True)
        else:
            # Show spinner while researching
            with st.spinner("🔍 Conducting deep research... This may take a moment..."):
                # Use settings from session state or defaults
                search_depth_val = st.session_state.get('search_depth_select', 'advanced')
                max_results_val = st.session_state.get('max_results_slider', 10)
                
                result = perform_research(topic, search_depth_val, max_results_val)
            
            # Store result
            st.session_state.current_result = result
            
            # Add to history
            if result.success:
                history_item = {
                    'topic': result.topic,
                    'timestamp': result.timestamp,
                    'key_points_count': len(result.key_points),
                    'sources_count': len(result.sources),
                    'result': result
                }
                st.session_state.research_history.append(history_item)
    
    # Display results
    if st.session_state.current_result:
        st.divider()
        display_research_results(st.session_state.current_result)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🔍 Deep Research App | Powered by <a href="https://tavily.com" target="_blank">Tavily AI</a></p>
        <p style="font-size: 0.8rem;">Research smarter, not harder</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    import re  # Import re for validation
    main()
