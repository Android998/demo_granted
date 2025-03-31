from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
import os
import re
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.llms import Llama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

load_dotenv()

INDEX_PATH = os.getenv("INDEX_PATH")
DIRECTORY = "./data" # Directorio que contiene los documentos

embeddings = FastEmbedEmbeddings()

def clean_text(text):
    """
    Limpia el texto eliminando encabezados, pies de página, índices y otros elementos repetitivos.
    """
    # Lista de patrones a eliminar (puedes ampliarla según necesites)
    patterns = [
        r'BOLETÍN OFICIAL DEL ESTADO',
        r'Núm\.\s*\d+',
        r'Pág\.\s*\d+',
        r'Sec\.\s*\w+'
    ]
    
    # Elimina cada patrón del texto
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Elimina líneas vacías y reduce espacios múltiples
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Parámetro: límite máximo de tokens para cada chunk (ajústalo según tu LLM)
max_tokens_limit = 1000

# Configura el LLM para segmentación semántica (puedes cambiar a Mixtra si lo integras)
llm = ChatGroq(temperature=0,model_name="llama3-70b-8192", max_tokens=8190)  # Reemplaza con la ruta de tu modelo

# Define el prompt para que el LLM separe el texto en segmentos semánticos
prompt_template = """
Dado el siguiente texto, divídelo en segmentos semánticamente coherentes que no superen {max_tokens} tokens cada uno.
Incluye un solapamiento de 200 tokens entre segmentos consecutivos para preservar el contexto.
Devuelve los segmentos, cada uno en una nueva línea.
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
    segments = [seg.strip() for seg in result.split("\n") if seg.strip()]
    return segments


all_docs = []
print("---------- INICIO ---------------")


# Itera sobre todos los archivos en el directorio
for filename in os.listdir(DIRECTORY):
    file_path = os.path.join(DIRECTORY, filename)
    
    # Verifica que sea un archivo
    if os.path.isfile(file_path):
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
                
                # Limpia el texto
                texto_limpio = clean_text(contenido)

                # Utiliza el LLM para segmentar el texto en chunks semánticos
                segmentos = semantic_chunking(texto_limpio, max_tokens_limit)
                all_docs.extend(segmentos)
                print(f"\nArchivo: {file_path}")
                print(f"Número de fragmentos generados: {len(segmentos)}")
                for i, doc in enumerate(docs):
                    print(f"Fragmento {i + 1}: {doc.page_content[:30]}...")  # Muestra los primeros 100 caracteres de cada fragmento

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