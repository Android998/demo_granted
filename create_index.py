import os
import re
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

INDEX_PATH = os.getenv("INDEX_PATH")
DIRECTORY = "./data" # Directorio que contiene los documentos

embeddings = FastEmbedEmbeddings()

# Parámetro: límite máximo de tokens para cada chunk (ajústalo según tu LLM)
max_tokens_limit = 1000

# Configura el LLM para segmentación semántica (puedes cambiar a Mixtra si lo integras)
llm = ChatGroq(temperature=0,model_name="llama3-70b-8192", max_tokens=8190)  # Reemplaza con la ruta de tu modelo

# Define el prompt para que el LLM separe el texto en segmentos semánticos
prompt_template = """
A continuación se te proporciona un texto extenso. Tu tarea es dividirlo en segmentos que sean semánticamente coherentes y que no excedan {max_tokens} tokens cada uno. Es muy importante que:
- Solo separes el texto en diferentes segmentos si estás absolutamente seguro de que corresponden a secciones o contextos diferentes.
- Evites fragmentar el texto innecesariamente: si el contenido es continuo y coherente, debe quedar en un solo segmento.
- Cada segmento debe incluir un solapamiento de 200 tokens con el siguiente, para preservar el contexto.
- No agregues ningún comentario, explicación o texto adicional. Devuelve únicamente la lista de segmentos.
- Para separar cada segmento en la respuesta, utiliza exactamente el delimitador '|*|'.

Por favor, devuelve la respuesta como una única cadena de texto en la que los segmentos estén concatenados, separados únicamente por '|*|'.

Texto:
{text}
"""

prompt = PromptTemplate(
    input_variables=["text", "max_tokens"],
    template=prompt_template
)

# Crea la cadena que utilizará el LLM para la segmentación
semantic_chain = LLMChain(llm=llm, prompt=prompt)

def semantic_chunking(text, max_tokens):
    """
    Utiliza el LLM para dividir el texto en segmentos semánticos que respeten el límite de tokens.
    Se espera que el LLM devuelva los segmentos separados por saltos de línea.
    """
    result = semantic_chain.run({"text": text, "max_tokens": max_tokens})
    # Separa los segmentos y elimina entradas vacías
    segments = [seg.strip() for seg in result.split("|*|") if seg.strip()]
    return segments

# Inicializa el tokenizer (ajusta el nombre del modelo según lo que utilices)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

def count_tokens(text: str) -> int:
    # Cuenta el número de tokens utilizando el tokenizer
    return len(tokenizer.encode(text))

# Define un splitter básico para pre-dividir documentos extensos
pre_splitter = RecursiveCharacterTextSplitter(
    chunk_size=6000,       # Ajusta según lo que consideres razonable (puedes usar caracteres o tokens)
    chunk_overlap=500,     # Un pequeño solapamiento para no perder contexto entre estos cortes
    length_function=count_tokens    
)

# Función para pre-dividir documentos largos
def pre_split_document(text):
    return pre_splitter.split_text(text)


all_docs = []
print("---------- INICIO ---------------")

# Itera sobre todos los archivos en el directorio
for filename in os.listdir(DIRECTORY):
    file_path = os.path.join(DIRECTORY, filename)
    
    # Verifica que sea un archivo
    if os.path.isfile(file_path):
        print(f"\nArchivo: {file_path}")
        try:
             # Detect file type and load it appropriately
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                continue  # Ignore unsupported formats
            
            # Carga el documento (sin aplicar un splitter previo)
            docs = loader.load()
            for doc in docs:
                # Obtén el contenido (usando 'page_content' si está disponible)
                contenido = doc.page_content if hasattr(doc, 'page_content') else doc

                # Pre-dividir el documento si es muy largo
                if len(contenido) > 3000:  # o ajusta esta condición según el conteo de tokens
                    fragmentos_preliminares = pre_split_document(contenido)
                else:
                    fragmentos_preliminares = [contenido]
                
                print(f"Dividido el texto en {len(fragmentos_preliminares)} presegmentos...")
                
                # Para cada fragmento preliminar, aplica el chunking semántico con el LLM
                for i, fragmento in enumerate(fragmentos_preliminares):
                    # Utiliza el LLM para segmentar el texto en chunks semánticos
                    segmentos = semantic_chunking(fragmento, max_tokens_limit)
                    print(f"Se han generado {len(segmentos)} segmentos...")
                    all_docs.extend(segmentos)
                    print(f"Número de fragmentos generados: {len(segmentos)}")
                    for seg in segmentos:
                        print(f"Fragmento: {seg[:100]}...") # Muestra los primeros 100 caracteres de cada fragmento

        except Exception as e:
            print(f"Error al cargar el archivo {file_path}: {e}")

# Imprime la cantidad total de documentos cargados
print(f"\n--> Se han generado {len(all_docs)} documentos en total.")

if os.path.exists(INDEX_PATH):
    vectorstore = Chroma(
    persist_directory=INDEX_PATH,
    embedding_function=embeddings
)
    print(f"--> Vectorstore cargado de {INDEX_PATH}")
else:
    
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=INDEX_PATH
        )
    print(f"\nSe ha creado un nuevo vectorstore {INDEX_PATH}\n")
    
print(f"\n---------- FIN ---------------")