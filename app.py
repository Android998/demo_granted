import streamlit as st
from langchain_utils import invoke_chain

# Set page configuration
st.set_page_config(page_title='Asistente Kit Digital', initial_sidebar_state='collapsed')

st.title("🤖 Asistente Kit Digital")

def render_or_update_model_info(model_name):
    """
    Renders or updates the model information on the webpage.

    Args:
        model_name (str): The name of the model.

    Returns:
        None
    """
    with open("./design/styles.css") as f:
        css = f.read()
    st.markdown('<style>{}</style>'.format(css), unsafe_allow_html=True)

    with open("./design/content.html") as f:
        html = f.read().format(model_name)
    st.markdown(html, unsafe_allow_html=True)

# Reset chat history
def reset_chat_history():
    """
    Resets the chat history by clearing the 'messages' list in the session state.
    """
    if "messages" in st.session_state:
        st.session_state.messages = []

model_options = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
max_tokens = {
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32768,
    "gemma-7b-it": 8192
}

# Initialize model
if "model" not in st.session_state:
    st.session_state.model = model_options[0]
    st.session_state.temperature = 0
    st.session_state.max_tokens = 8192

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar configuration
with st.sidebar:
    st.title("Configuración de modelo")

    # Select model
    st.session_state.model = st.selectbox(
        "Elige un modelo:",
        model_options,
        index=0
    )

    # Select temperature
    st.session_state.temperature = st.slider('Selecciona una temperatura:', min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

    # Select max tokens
    if st.session_state.max_tokens > max_tokens[st.session_state.model]:
        max_value = max_tokens[st.session_state.model]

    st.session_state.max_tokens = st.number_input('Seleccione un máximo de tokens:', min_value=1, max_value=max_tokens[st.session_state.model], value=max_tokens[st.session_state.model], step=100)

    # Reset chat history button
    if st.button("Vaciar Chat"):
        reset_chat_history()

# Render or update model information
render_or_update_model_info(st.session_state.model)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "figure" in message["aux"].keys() and len(message["aux"]["figure"]) > 0:
            st.plotly_chart(message["aux"]["figure"][0])
        st.text("")

# Accept user input
prompt = st.chat_input("Escribe tu pregunta sobre el Kit Digital...")

if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        response = invoke_chain(
            question=prompt,
            messages=st.session_state.messages,
            model_name=model_options[model_options.index(st.session_state.model)],
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens
        )
        st.write(response)
        if "figure" in invoke_chain.aux.keys() and len(invoke_chain.aux["figure"]) > 0:
            st.plotly_chart(invoke_chain.aux["figure"][0])
        if hasattr(invoke_chain, 'recursos'):
            for recurso in invoke_chain.recursos:
                st.button(recurso)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "aux": {}})
    st.session_state.messages.append({"role": "assistant", "content": invoke_chain.response, "aux": invoke_chain.aux})