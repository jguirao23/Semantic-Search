import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import time
import numpy as np
from PyPDF2 import PdfReader

# Loading the text
pdfFileObj = open('PSD2.pdf', 'rb')
pdfReader = PdfReader(pdfFileObj)

content = ""
for page_number in range(0,len(pdfReader.pages)):
  page = pdfReader.pages[page_number]
  content += page.extract_text()

print(content)
pdfFileObj.close()

# Semantic Search
## Separating the text in sentences, that will be encoded later for semantic search
nlp = spacy.load('en_core_web_md')
doc = nlp(content)
for sent in doc.sents:
  print(sent.text)

## Saving the sentences in a DataFrame
sentence_df = pd.DataFrame([sent.text for sent in doc.sents], columns=['sentence'])
sentence_df.head()
sentence_df.tail()

## We need to check if the sentences are under the maximum length allowed by the model (512):
sentence_df['sent_len'] = sentence_df['sentence'].apply(lambda x: len(x.split(" ")))
sentence_df['sent_len'].value_counts().sort_index(ascending=False)

## Text embedding
model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
embeddings = model.encode([sent.text for sent in doc.sents])
len(embeddings)
len(embeddings[0])
print(embeddings[0])

## Entering FAISS
encoded_data = np.array(embeddings.astype('float32'))
index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
index.add_with_ids(encoded_data, np.array(range(0, len(embeddings))))
faiss.write_index(index, 'doc.index')

def fetch_sentence(dataframe_idx):
    """ 
    This function retrieves the sentence corresponding to the desired Index ID.
    It will be used inside the search funcion.
    """    
    sentence = sentence_df.loc[dataframe_idx, 'sentence']
    return sentence

def search(query, top_k, index, model):
  """
  This model returns the top K sentences related to the query.
  """
  t=time.time()
  query_vector = model.encode([query])
  top_k = index.search(query_vector, top_k)
  print('>>>> Results in Total Time: {}'.format(time.time()-t))
  top_k_ids = top_k[1].tolist()[0]
  top_k_ids = list(np.unique(top_k_ids))
  results = [fetch_sentence(idx) for idx in top_k_ids]
  return results
 

# Streamlit app
st.title(":mag: Semantic Search")
st.divider()
query = st.text_input(label=':blue[Query]', placeholder = 'Enter your question here')
results = search(query, top_k=1, index=index, model=model)

if query == "":
  st.write("")
else:
  st.write(results)
