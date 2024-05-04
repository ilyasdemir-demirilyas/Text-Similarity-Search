# Text-Similarity-Search

```markdown
# Text Similarity Search Application

This application is a Streamlit-based tool that allows users to perform a similarity search by uploading a text file and entering a query. It utilizes TextProcessor and VectorSearch classes to process the uploaded text and perform the similarity search using OpenAI Embeddings.

## Installation

To install the required dependencies, you can use pip and the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

### Requirements

Here are the required packages and their versions:

```
streamlit==1.34.0
langchain-community==0.0.36
langchain-openai==0.1.6
langchain-text-splitters==0.0.1
langchain-chroma==0.1.0
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/ilyasdemir-demirilyas/Text-Similarity-Search.git
```

2. Navigate to the project directory:

```bash
cd Text-Similarity-Search
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run Search_web_page.py
```

5. Upload a text file and enter a query to perform a similarity search.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

Chroma is an AI-native open-source vector database focused on developer productivity and happiness. Chroma is licensed under the Apache 2.0 License.
