from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv
import os
from functools import lru_cache

load_dotenv()

INDEX_PATH = os.getenv("INDEX_PATH")

@lru_cache(maxsize=None)
def get_model(model_name, temperature, max_tokens):
    """
    Returns a language model based on the specified model name, temperature, and max tokens.

    Args:
        model_name (str): The name of the language model.
        temperature (float): The temperature parameter for generating responses.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        ChatGroq: The language model object based on the specified parameters.
    """
    print(f"Parámetros de modelo {model_name, temperature, max_tokens}")
    llm = {
        "llama3-70b-8192": ChatGroq(temperature=temperature,model_name="llama3-70b-8192", max_tokens=max_tokens),
        "llama3-8b-8192": ChatGroq(temperature=temperature,model_name="llama3-8b-8192", max_tokens=max_tokens),
        "mixtral-8x7b-32768": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
        "gemma-7b-it": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
    }
    return llm[model_name]


embeddings = FastEmbedEmbeddings()
vectorstore = Chroma(
    persist_directory=INDEX_PATH,
    embedding_function=embeddings
    )
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 2
        }
    )

# First we need a prompt that we can pass into an LLM to generate this search query

prompt_query = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            """Dado el contenido anterior, genera una consulta de búsqueda para obtener información relevante para la conversación.\
               La consulta debe utilizar palabras clave para que se entienda la esencia del mensaje.
            """,
        ),
    ]
)

prompt_main = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente virtual centrado en el mundo de la cocina. Tu especilidad es ayudar a cocineros a realizar sus recetas.
            Te conocen como Monty, el Chef.
            Responde a la pregunta del usuario utilizando ÚNICAMENTE el contexto que tienes a continuación:\n\n{context}.
            
            Solo debes incluir información que aparece en el contexto. En caso de no disponer de la información indica que no dispones de los datos suficientes.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)

@lru_cache(maxsize=None)
def get_rag_chain(model_name, temperature, max_tokens):
    """
    Create and return a retrieval-augmented generation (RAG) chain.

    Args:
        model_name (str): The name of the model to use.
        temperature (float): The temperature parameter for generation.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        rag_chain: The retrieval-augmented generation chain.

    """
    model = get_model(model_name, temperature, max_tokens)
    
    retriever_chain = create_history_aware_retriever(model, retriever, prompt_query)
    document_chain = create_stuff_documents_chain(model, prompt_main)

    rag_chain = create_retrieval_chain(retriever_chain, document_chain)

    return rag_chain


def create_history(messages):
    """
    Creates a ChatMessageHistory object based on the given list of messages.

    Args:
        messages (list): A list of messages, where each message is a dictionary with "role" and "content" keys.

    Returns:
        ChatMessageHistory: A ChatMessageHistory object containing the user and AI messages.

    """
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_chain(question, messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192):
    model = get_model(model_name, temperature, max_tokens)

    # 🔹 Inicialización de variables
    response = ""
    aux = {}
    
    # 🔹 Detectar intención y generar respuesta
    detected_response = detect_intent_with_history(question, messages, model)
    print(f"🔍 Detected Intent: {detected_response}")

    # Handle response based on intent
    if "Informacion general de funcionalidades del agente" in detected_response:
        print("hola")
        # Llamamos a la cadena que describe las funcionalidades del bot
        response = agent_description_chain(model)

    elif "Información sobre el Kit Digital" in detected_response:
        # Paso (A): Buscar documentos usando SOLO la pregunta
        docs = get_docs_for_question(question, k=4)
        # Paso (B): Generar respuesta final con LLMChain
        response = generate_final_answer_llmchain(question, messages, docs, model)

    elif "Generación de documentos" in detected_response:
        response = detected_response

    elif "Automatización del formulario online" in detected_response:
        response = detected_response

    else:
        response = detected_response


    # 🔹 Guardar en la historia del chat (manteniendo coherencia con la estructura original)
    invoke_chain.response = response
    invoke_chain.history = messages  # No modificamos mensajes aquí, solo lo mantenemos por compatibilidad
    invoke_chain.aux = aux

    return response



def get_docs_for_question(question, k=5):
    """
    Devuelve los 'k' documentos más relevantes 
    usando vectorstore.similarity_search(question).
    """
    results = vectorstore.similarity_search(question, k=k)
    return results


def limit_history(messages, max_messages=3):
    # Si el historial es más corto o igual al máximo permitido,
    # no necesitamos recortar.
    if len(messages) <= max_messages:
        return messages
    else:
        # Devolvemos sólo los últimos 'max_messages'.
        return messages[-max_messages:]

def detect_intent_with_history(question, messages, model):
    """
    Clasifica la intención del usuario, tomando en cuenta:
    - Parte del historial de chat
    - La última pregunta del usuario

    Devuelve SOLO la categoría de intención en texto plano.
    """
    # 1) Recortar el historial
    limited_history = limit_history(messages, max_messages=4)

    # 2) Convertir la parte de historial en un string
    hist_str = ""
    for m in limited_history:
        role = m["role"]  # 'user' o 'assistant'
        content = m["content"]
        hist_str += f"[{role.upper()}]: {content}\n"

    # 3) Contenido para el system
    system_template = """
                    Eres un asistente virtual especializado en ayudar a pequeñas y medianas empresas (PYMEs) con la solicitud de la subvención del Kit Digital en España. 
                    Tu objetivo es guiar a los usuarios en tres áreas clave:

                    1️⃣ **Información sobre el Kit Digital**: Responder preguntas sobre requisitos, proceso de solicitud, documentación, plazos, etc.
                    2️⃣ **Generación de documentos**: Ayudar a crear declaraciones responsables, formularios y otros documentos necesarios.
                    3️⃣ **Automatización del formulario online**: Asistir en la cumplimentación automática de formularios en la Sede Electrónica de Red.es.

                    Si la consulta del usuario no está relacionada con estas áreas, debes indicar que no puedes ayudar con ese tema.
                    """

    # 4) Contenido para el mensaje del 'user'
    #    Incluimos el historial previo y la última pregunta
    user_template = """
                    Esta es la conversación previa (limitada):
                    {historia}

                    La última pregunta (mensaje) del usuario actual es: 
                    {question}

                    Clasifica la intención del usuario en una de las siguientes categorías:
                    - Informacion general de funcionalidades del agente
                    - Información sobre el Kit Digital
                    - Generación de documentos
                    - Automatización del formulario online
                    - Otra (si la consulta no está dentro de tus capacidades)

                    Responde UNICAMENTE con la categoria exacta
                    """

    # 5) Crear un ChatPromptTemplate con 2 mensajes: System y Human
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ])

    # 6) Crear un LLMChain con el modelo y el prompt
    chain = LLMChain(llm=model, prompt=chat_prompt)

    # 7) Ejecutar la cadena, pasando las variables del template
    result = chain.run(historia=hist_str, question=question)

    # Devuelve la clasificación limpia (y quita espacios)
    return result.strip()


def agent_description_chain(model):
    # Mensaje de 'system', con contexto e instrucciones
    system_template = """
                    Eres un asistente especializado en guiar a las PYMEs españolas 
                    en la subvención del Kit Digital.

                    Tu misión:
                    1) Responder dudas sobre requisitos, proceso de solicitud, plazos, etc.
                    2) Ayudar a crear documentos (declaraciones responsables, formularios...).
                    3) Asistir en la automatización del formulario online en la Sede Electrónica.

                    Cuando se te solicite "Información general de funcionalidades", 
                    debes describir de forma clara qué sabes hacer.
                    """

    # Mensaje del 'user', pidiendo la descripción
    user_template = """
                    Por favor, proporciona una descripción general de tus funcionalidades 
                    y de qué maneras puedes ayudar al usuario con el Kit Digital.
                    """

    # Creamos el prompt en formato “chat”
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ])

    # Creamos la cadena LLMChain con el modelo y el prompt
    chain = LLMChain(llm=model, prompt=chat_prompt)

    # Ejecutamos la cadena. No usamos variables, así que:
    description = chain.run({})
    # O bien:
    # description = chain.predict()

    return description


def generate_final_answer_llmchain(question, messages, docs, model):
    # 1) Recortar historial
    limited_history = limit_history(messages, max_messages=6)
    
    # 2) Unir documentos en un bloque de texto
    doc_content = "\n\n".join(doc.page_content for doc in docs)
    
    # 3) Convertir el historial a texto
    hist_str = ""
    for m in limited_history:
        role = m["role"]  # 'user' o 'assistant'
        content = m["content"]
        hist_str += f"[{role.upper()}]: {content}\n"

    # 4) Definir las instrucciones de 'system'
    system_instructions = (
        "Eres un asistente experto en el Kit Digital. "
        "Responde de forma coherente, teniendo en cuenta la conversación previa. "
        "Si no encuentras la información en los documentos, di que no hay datos suficientes."
    )

    # 5) Crear el contenido del 'user'
    #    Va a incluir: historial recortado, documentos relevantes y la pregunta final
    user_template = """CONVERSACIÓN HASTA AHORA:
                    {hist_str}

                    DOCUMENTOS RELEVANTES:
                    {doc_content}

                    PREGUNTA ACTUAL DEL USUARIO:
                    {question}
                    """

    # 6) Crear un ChatPromptTemplate con DOS mensajes: System y Human
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_instructions),
        HumanMessagePromptTemplate.from_template(user_template)
    ])

    # 7) Crear la cadena LLMChain con ese prompt y tu LLM (ChatGroq)
    chain = LLMChain(llm=model, prompt=chat_prompt)

    # 8) Ejecutar la cadena pasando los argumentos que aparecen en el template
    response = chain.run(
        hist_str=hist_str, 
        doc_content=doc_content, 
        question=question
    )

    return response
