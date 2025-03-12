from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

INDEX_PATH = os.getenv("INDEX_PATH")
DIRECTORY = "./data" # Directorio que contiene los documentos

embeddings = FastEmbedEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=0
)

# Inicializa una lista vacía para almacenar los documentos
all_docs = []
print(f"---------- INICIO ---------------")
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
            
            # Aplica el text_splitter al documento
            docs = loader.load_and_split(text_splitter=text_splitter)
            
            # Agrega los documentos cargados a la lista de todos los documentos
            all_docs.extend(docs)
            
            # Imprime información sobre los documentos cargados
            print(f"\nArchivo: {file_path}\n")
            print(f"Número de fragmentos generados: {len(docs)}")
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