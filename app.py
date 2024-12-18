import streamlit as st
from aiers import AIERS

# Define the main function
def main():
    # Page Configurations
    st.set_page_config(
        page_title="AIERS Emergency Response System",
        page_icon="🚨",
        layout="centered",
    )

    # Initialize AIERS system
    if 'ai' not in st.session_state:
        st.session_state.ai = AIERS()

    if 'initialized' not in st.session_state:
        st.session_state.ai.conversation.predict(input=st.session_state.ai.system_prompt)
        st.session_state.initialized = True

    # Custom CSS for modern design
    st.markdown(
        """
        <style>
        /* Main Container */
        .main {
            background-color: #2e3350; /* Deep Purple */
            color: #ffffff; /* White Text */
            padding: 10px;
        }

        /* Global Styles */
        .stApp {
            background-color: #2e3350; /* Deep Purple */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #f7b632; /* Yellow */
        }
        p, div, .markdown-text-container {
            font-family: 'Arial', sans-serif;
            color: #ffffff; /* White Text */
        }

        /* Sidebar Styling */
        .css-1d391kg {
            background-color: #5a87bb !important; /* Blue */
        }
        .css-1d391kg h2 {
            color: #ffffff; /* White */
        }
        .css-1d391kg .st-radio-label {
            color: #ffffff !important; /* White */
        }

        /* Button Styling */
        .custom-button {
            background-color: #f26640; /* Orange */
            color: #ffffff; /* White Font */
            font-size: 1rem;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            margin: 10px 0;
            transition: 0.3s ease-in-out;
            cursor: pointer;
        }
        .custom-button:hover {
            background-color: #f7b632; /* Yellow */
            transform: scale(1.05);
        }

        /* Highlight Boxes */
        .highlight {
            background-color: #5a87bb; /* Blue */
            border: 2px solid #f26640; /* Orange */
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            color: #ffffff; /* White Text */
        }

        /* Header Styling */
        .header-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: 700;
            color: #f7b632; /* Yellow */
            margin-bottom: 15px;
        }
        .divider {
            height: 2px;
            background-color: #f26640; /* Orange */
            border: none;
            margin: 20px 0;
        }

        /* Input Box Styling */
        textarea, select, .stTextInput>div>input {
            background-color: #2e3350; /* Deep Purple */
            color: #ffffff; /* White Text */
            border: 1px solid #f7b632; /* Yellow Border */
            border-radius: 8px;
            padding: 10px;
            font-family: 'Arial', sans-serif;
        }
        textarea:focus, select:focus, .stTextInput>div>input:focus {
            border: 1px solid #f26640; /* Orange Focus Border */
            outline: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # App Title
    st.markdown("<div class='header-title'>AIERS Emergency Response System</div>", unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.header("🚨 Navigation")
    nav_option = st.sidebar.radio("Select a Section:", ["Home", "Report an Emergency", "Emergency History", "About AIERS"])

    if nav_option == "Home":
        display_home()
    elif nav_option == "Report an Emergency":
        display_report_emergency()
    elif nav_option == "Emergency History":
        display_emergency_history()
    elif nav_option == "About AIERS":
        display_about()

# Home Page
def display_home():
    st.subheader("Welcome to AIERS")
    st.markdown(
        """
        AIERS is your **AI-powered emergency response assistant**, designed to assist emergency responders and individuals in critical situations. Key features include:
        - 🚨 **Speech-to-Text Emergency Reporting**
        - 📢 **Automated Text-to-Speech Guidance**
        - 🔍 **Real-time Emergency Classification**
        - 🙌 **Caller Reassurance and Guidance**
        """
    )

# Report Emergency Page
def display_report_emergency():
    st.subheader("Report an Emergency")
    method = st.radio("Select Input Method:", ("Speech", "Custom Text", "Category"), key="input_method")

    if method == "Speech":
        st.markdown("<div class='highlight'>🎙️ Record your emergency report by speaking. AIERS will transcribe and process it.</div>", unsafe_allow_html=True)
        if st.button("🔴 Start Recording"):
            try:
                # Record audio
                audio_filename = "recorded_audio.wav"
                st.session_state.ai.record_audio(audio_filename, duration=6)
            
                # Transcribe the recorded audio
                transcript = st.session_state.ai.transcribe_audio(audio_filename)
            
                # Identify role (dispatcher or caller)
                role = st.session_state.ai.determine_role(transcript)
            
                # Display transcription
                st.subheader("Your Speech")
                st.text_area("Transcribed Text:", value=transcript, height=100)
            
                # Get LLM response based on role
                ai_response = st.session_state.ai.get_llm_response(transcript, role)
            
                # Display AI response (adjust based on role)
                st.subheader(f"AI Response for {role.capitalize()}")
                st.write(ai_response)
            
                # Generate and play TTS for the identified role
                st.session_state.ai.text_to_speech(ai_response, role)
            
            except Exception as e:
                st.error(f"Error in conversation process: {str(e)}")

    elif method == "Custom Text":
        st.markdown("<div class='highlight'>✏️ Type out the details of the emergency and submit them for processing.</div>", unsafe_allow_html=True)
        custom_text = st.text_area("Describe the emergency:")
        if st.button("Submit Custom Text"):
            # Identify role based on custom text
            role = st.session_state.ai.determine_role(custom_text)
            ai_response = st.session_state.ai.get_llm_response(custom_text, role)
            
            # Display AI response
            st.subheader(f"AI Response for {role.capitalize()}")
            st.write(ai_response)
            st.session_state.ai.text_to_speech(ai_response, role)

    elif method == "Category":
        st.markdown("<div class='highlight'>📂 Select a category and subcategory to generate an emergency report.</div>", unsafe_allow_html=True)
        category = st.selectbox("Main Category", ("Accident", "Health", "Disaster", "Robbery"))
        subcategories = {
            "Accident": ["Traffic Accident", "Workplace Accident", "Home Injury", "Fire Incident", "Electrical Shock"],
            "Health": ["Heart Attack", "Stroke", "Severe Allergic Reaction", "Breathing Difficulty", "High Fever", "Poisoning", "Unconsciousness"],
            "Disaster": ["Earthquake", "Flood", "Landslide", "Tornado", "Wildfire", "Tsunami", "Volcanic Eruption"],
            "Robbery": ["Home Invasion", "Armed Robbery", "Pickpocketing", "Vehicle Theft", "Store Robbery", "Cybercrime"],
        }
        subcategory = st.selectbox("Subcategory", subcategories[category])
        predefined_text = f"This is a {subcategory.lower()} under the category of {category.lower()}. Immediate assistance is required."
        if st.button("Submit Category Emergency"):
            # Identify role for predefined category
            role = "dispatcher"  # Always dispatcher for category-based emergencies
            ai_response = st.session_state.ai.get_llm_response(predefined_text, role)
                    
            # Display AI response
            st.subheader(f"AI Response for {role.capitalize()}")
            st.write(ai_response)

            # Generate and play TTS for the dispatcher
            st.session_state.ai.text_to_speech(ai_response, role)

# Emergency History Page
def display_emergency_history():
    st.subheader("Emergency History")
    st.markdown("View your recent emergency submissions and their responses.")
    st.warning("🛠️ Feature under development")

# About Page
def display_about():
    st.subheader("About AIERS")
    st.markdown(
        """
        AIERS (**AI Emergency Response System**) is built to revolutionize emergency services by using state-of-the-art AI technologies, including:
        - 🗣️ **Speech-to-Text Transcription (Silero)**
        - 📢 **Text-to-Speech Guidance (Microsoft Edge TTS)**
        - 🤖 **Natural Language Understanding (Google Generative AI)**
        - 🚦 **Emergency Routing Intelligence**

        Developed with the vision to enhance emergency response efficiency and provide critical guidance during emergencies.
        """
    )

# Emergency Processing
def process_emergency_report(report_text):
    try:
        result = st.session_state.ai.process_emergency(report_text)

        if result:
            st.success("Emergency report processed successfully.")
            st.subheader("🚨 Dispatcher Response")
            st.code(result['dispatcher_routing_response'], language="text")
            st.subheader("📞 Caller Guidance")
            st.code(result['caller_routing_response'], language="text")
            st.subheader("🤖 AI Detailed Analysis")
            st.write(result['llm_response'])
    except Exception as e:
        st.error(f"Error processing emergency: {e}")

if __name__ == "__main__":
    main()
