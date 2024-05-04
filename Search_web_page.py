import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import streamlit as st
import tempfile


#Kod Açıklaması :
#Bu kod, kullanıcının bir metin dosyası yükleyerek ve bir sorgu girerek benzerlik araması yapmasını sağlayan bir
#Streamlit uygulamasıdır.

#Kod, TextProcessor ve VectorSearch sınıflarını içerir. TextProcessor sınıfı, yüklenen metin dosyasını işlemek ve
#geçici bir dosyaya yazmak için kullanılır. VectorSearch sınıfı, işlenen metinleri alır, belirli bir sorguya göre
#benzerlik araması yapmak için OpenAI Embeddings kullanır.

#Kod ayrıca Streamlit arayüzü sağlar. Kullanıcı, bir metin dosyası yükleyebilir ve bir sorgu girebilir. Ardından,
#"Ara" düğmesine tıkladıklarında, sorgularıyla benzerlik araması yapılır ve sonuçları ekranda gösterilir.

#Kodun kullanımıyla ilgili açıklamalar ve hata mesajları, kullanıcıya yönlendirme ve bilgilendirme sağlar.

# Code Description:
# This code is a Streamlit application that allows users to upload a text file and perform a similarity search by entering a query.

# The code includes TextProcessor and VectorSearch classes. The TextProcessor class is used to process the uploaded text file and write it to a temporary file. The VectorSearch class takes the processed text and uses OpenAI Embeddings to perform a similarity search based on a specific query.

# The code also provides a Streamlit interface. Users can upload a text file and enter a query. When they click the "Search" button, a similarity search is performed with their queries and the results are displayed on the screen.

# The code provides explanations and error messages for usage to guide and inform the user.


class TextProcessor:
    def __init__(self, file_content):
        self.file_content = file_content

    def load_and_process_text(self):
        try:
            # Write the file content to a temporary file
            temp_file_path = self.write_temp_file()
            # Load the temporary file with TextLoader
            raw_documents = TextLoader(temp_file_path, encoding='utf-8').load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            self.documents = text_splitter.split_documents(raw_documents)
            return True
        except Exception as e:
            print("Error:", e)
            return False

    def write_temp_file(self):
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, "uploaded_file.txt")
        # Write the file content
        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(self.file_content)
        return temp_file_path


class VectorSearch:
    def __init__(self, documents):
        self.documents = documents

    def perform_similarity_search(self, query):
        try:
            db = Chroma.from_documents(self.documents, OpenAIEmbeddings())
            embedding_vector = OpenAIEmbeddings().embed_query(query)
            docs = db.similarity_search_by_vector(embedding_vector)
            return docs[0].page_content
        except Exception as e:
            print("Error:", e)
            return None

st.title("Text Similarity Search")

# Upload the text file
uploaded_file = st.file_uploader("Choose a file")

# User query
query = st.text_input("Please enter a query:")

# Search for similarity button
if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    elif uploaded_file is not None:
        try:
            # Text processing
            file_content = uploaded_file.getvalue().decode("utf-8")

            # Process the file
            text_processor = TextProcessor(file_content)
            processed_text = text_processor.load_and_process_text()

            if text_processor.load_and_process_text():
                # Check for empty text file
                if not text_processor.documents:
                    st.warning("The uploaded text file is empty. Please upload a valid text file.")
                else:
                    vector_search = VectorSearch(text_processor.documents)
                    result = vector_search.perform_similarity_search(query)
                    if result:
                        st.text_area("Similarity Search Result", value=result, height=300)
                    else:
                        st.warning("No similarity found.")
            else:
                st.error("Failed to upload the text file.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please upload a text file.")




