# uvicorn fastapi_test:app --reload
from fastapi import FastAPI, HTTPException, File, UploadFile
from sqlalchemy import create_engine, MetaData, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy import DateTime, func
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx
from sqlalchemy import text
from datetime import datetime
from spacy.lang.en.examples import sentences 
from keras.preprocessing.text import Tokenizer
#from sqlalchemy import declarative_base
from sqlalchemy import LargeBinary
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse,FileResponse
#from sqlalchemy.orm import from_orm
#from modeling import realtime_pred #, from_image_to_bytes
#from decoding import from_image_to_bytes
import spacy #advanced Natural Language Processing in Python and Cython
import pytextrank #ranking text
from pprint import PrettyPrinter #print in a pretty way 
from PIL import Image
import re
import base64
import io

import requests
import shutil
import pickle
import json, pandas as pd
import numpy as np
import os
from fastapi import Form, UploadFile
import json
from sqlalchemy.orm import Session
from fastapi import Depends

import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from keras.datasets import reuters
from keras import datasets
from keras.preprocessing.sequence import pad_sequences

import torch
from diffusers import StableDiffusionPipeline


import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
#import modeling
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM
from fastapi_cors import CORS


# FastAPI 애플리케이션 생성
app = FastAPI()
# CORS 설정 추가
#origins = ["*"]

origins = [
    "https://ml-serving-api-u2.fly.dev/",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5500",# FastAPI 서버 주소
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def from_image_to_bytes(img):
   
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format=img.format)
    imgByteArr = imgByteArr.getvalue()
    encoded = base64.b64encode(imgByteArr)
    decoded = encoded.decode('ascii')
    return decoded


DATABASE_URL = "mysql+pymysql://root:{}@localhost:{}}/news"
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class News(Base):
    __tablename__ = "label_test"
    news_title = Column(String,primary_key=True, index=True)
    news_description = Column(String, index=True, default=None)    
    
class Nation(Base):
    __tablename__ = "country_abbreviation"
    country = Column(String,primary_key=True, index=True)
    abbreviation = Column(String, index=True, default=None) 


class MapOneOutput(Base):
    __tablename__ = "country_title_mapping"
    id = Column(Integer,primary_key=True, index=True)
    title  = Column(String, index=True, default=None) 
    sorting = Column(String, index=True, default=None) 


title_vocab_size = 100
des_vocab_size = 200



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() # 안닫으면 ContextManager에 문제생김 
# 테이블 생성
Base.metadata.create_all(bind=engine) 

def do_query(query):
    session = SessionLocal()
    result = session.execute(query)
    session.commit()
    session.close()
    return result




# 테스트용 API 엔드포인트
@app.get("/test-database-connection")
async def test_database_connection():
    try:
        engine.connect()
        return {"message": "Database connection successful"}
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/news")
async def read_news():
    session = SessionLocal()
    news = session.query(News).all()
    #news.DataFrame()
    session.close()
    return news

@app.get("/nation")
async def read_nation():
    session = SessionLocal()
    nation = session.query(Nation).all()
    session.close()
    return nation

@app.get("/news_nation_mapping")
async def read_news():
    session = SessionLocal()
    #news = session.query(News).all()
    nation = session.query(Nation).all()
    def dd():
        for nation_filter in nation:
            map_country = nation_filter.country
            #map_abb = nation_filter.abbreviation # abbreviation 을 country 아래 한 칼럼으로 모두 합치는게 나음.
            #print(map_country) # interation checked
            for pp in map_country:#, map_abb:
                #print(pp)
                news_filter = session.query(News).filter(News.news_title.like('%pp%')).all()
        return news_filter
    session.close()
    return dd() # plotting to streamlit network map by nation 


# @app.put("/ranking_network")
# async def ranked_key_news():
#     session = SessionLocal()
#     news = session.query(News).all()
#     for abbsum in news:
#         sum = abbsum.news_description
#         #print(sum)
#         sentences = sent_tokenize(str(sum))
#         #print(sentences)
#         vocab = {}
#         preprocessed_sentences = []
#         stop_words = set(stopwords.words('english'))

#         for sentence in sentences:
#             # 단어 토큰화
#             tokenized_sentence = word_tokenize(sentence)
#             result = []

#             for word in tokenized_sentence:
#                 word = word.lower() .
#                 if word not in stop_words: 
#                     if len(word) > 1: 
#                         result.append(word)
#                         if word not in vocab:
#                             vocab[word] = 0
#                         vocab[word] += 1
#             preprocessed_sentences.append(result)
#         preprocessed_sentences = preprocessed_sentences[0]  # preprocessed_sentences
#         #print(preprocessed_sentences)

#         tokenizer = Tokenizer()
#         tokenizer.fit_on_texts(preprocessed_sentences)
#         vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True) # rankied keys from news
#         rankedkey = dict((x,y) for x,y in vocab_sorted)

#     session.close()
#     return rankedkey


## Ranking keywors for netwrok graph by keywords
@app.put("/ranking_network")
async def ranked_key_news():
    session = SessionLocal()
    news = session.query(News).all()
    #dd = pd.DataFrame(news)
    #print(news)
    for abbsum in news:
        sum = abbsum.news_description
        print("this is sum  " + sum)
        #sum = pd.DataFrame(sum)
        #print(sum)
        sentences = sent_tokenize(str(sum))
        #print(sentences)
        vocab = {}
        preprocessed_sentences = []
        stop_words = set(stopwords.words('english'))

        for sentence in sentences:
            # 단어 토큰화
            tokenized_sentence = word_tokenize(sentence)
            result = []

            for word in tokenized_sentence:
                word = word.lower() 
                if word not in stop_words: 
                    if len(word) > 2: 
                        result.append(word)
                        if word not in vocab:
                            vocab[word] = 0
                        vocab[word] += 1
            preprocessed_sentences.append(result)
        preprocessed_sentences = preprocessed_sentences[0]  # preprocessed_sentences
        print(preprocessed_sentences)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(preprocessed_sentences)
        vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True) # rankied keys from news
        rankedkey = dict((x,y) for x,y in vocab_sorted)

    session.close()
    return rankedkey

@app.put("/news_toss")
async def toss_news():
    session = SessionLocal()
    news = session.query(News).all()
    newsd = session.query(News.news_description).all()
    #dd = pd.DataFrame(news)
    print(newsd)
    return newsd

@app.put("/title_toss")
async def toss_title():
    session = SessionLocal()
    news = session.query(News).all()
    newsd = session.query(News.news_title).all()
    #dd = pd.DataFrame(news)
    print(newsd)
    return newsd


    # for abbsum in news:
    #     sum = abbsum.news_description
    #     print("this is toss  " + sum)
    #     return sum
## Rank Keywords
# @app.put("/ranking_network")
# async def ranked_key_news():
#     session = SessionLocal()
#     news = session.query(News).all()
#     print(news)
#     #damn = read_news()
#     for abbsum in news:
#         sum = abbsum.news_description
#         #sumt = abbsum.news_title
#         #sum = session.query(News).filter(News.news_description)
#         #news_df = pd.DataFrame(sum)
#         print(sum)

#         sentences = sent_tokenize(str(sum))

#         vocab = {}
#         preprocessed_sentences = []
#         stop_words = set(stopwords.words('english'))

#         for sentence in sentences:
#             # 단어 토큰화
#             tokenized_sentence = word_tokenize(sentence)
#             result = []

#             for word in tokenized_sentence:
#                 word = word.lower() 
#                 if word not in stop_words:
#                     if len(word) > 1: 
#                         result.append(word)
#                         if word not in vocab:
#                             vocab[word] = 0
#                         vocab[word] += 1
#             preprocessed_sentences.append(result)
#         preprocessed_sentences = preprocessed_sentences[0]  # preprocessed_sentences
#         tokenizer = Tokenizer()
#         tokenizer.fit_on_texts(preprocessed_sentences)
#         vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True) # rankied keys from news
#         rankedkey = dict((x,y) for x,y in vocab_sorted)

#     session.close()
#     return sum
# from langchain.llms import OpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.chains import SimpleSequentialChain
# llm = OpenAI(temperature=1)

pp = PrettyPrinter()
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textrank")

# def summary_for_article(num,prin=False):
    
#     ans = "" # collecting the summary from the generator
#     doc = nlp(newspd[num]) #apply the pipeline
    
#     for i in doc._.textrank.summary(limit_phrases=10, limit_sentences=1): #get the summary
#         ans+=str(i)
        
#     phrases_and_ranks = [ (phrase.chunks[0], phrase.rank) for phrase in doc._.phrases] # get important phrases
    
#     if prin: # print
#         print(newspd[num])
#         print("\n_______ to ______\n")
#         print(ans)
#         print("\n_______ important phrases ______\n")
#         pp.pprint(phrases_and_ranks[:10])
        
#     return ans

# @app.put("/summerization")
# async def summerize_news():
#     # session = SessionLocal()
#     # news = session.query(News).all()
#     # #damn = read_news()
#     # for abbsum in news:
#     #     sum = abbsum.news_description
#     #     realtime_pred(sum)
#     #     #print(pred)
#     # session.close()
#     session = SessionLocal()
#     news = session.query(News.news_description).first() #     newsd = session.query(News.news_description).all()
#     #newspd = pd.DataFrame(news)
#     #print(newspd)
    
    
#     return news

# ## Ranking keywors for netwrok graph by keywords
# @app.put("/news_summerization")
# async def summerizing_news():
#     session = SessionLocal()
#     news = session.query(News.news_description).all()
#     #dd = pd.DataFrame(news)
#     #print(news)
#     for abbsum in news:
#         sum = abbsum#.news_description
#         #print("this is sum  " + sum)
#         #sum = pd.DataFrame(sum)
#         #print(sum)
#         sentences = sent_tokenize(str(sum))
#         #print(sentences)
#         vocab = {}
#         preprocessed_sentences = []
#         stop_words = set(stopwords.words('english'))

#         for sentence in sentences:
#             # 단어 토큰화
#             tokenized_sentence = word_tokenize(sentence)
#             result = []

#             for word in tokenized_sentence:
#                 word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄인다.
#                 if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거한다.
#                     if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거한다.
#                         result.append(word)
#                         if word not in vocab:
#                             vocab[word] = 0
#                         vocab[word] += 1
#             preprocessed_sentences.append(result)
#         preprocessed_sentences = preprocessed_sentences[0]  # preprocessed_sentences
#         print(preprocessed_sentences)

#         tokenizer = Tokenizer()
#         tokenizer.fit_on_texts(preprocessed_sentences)
        
#         #vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True) # rankied keys from news
#         #rankedkey = dict((x,y) for x,y in vocab_sorted)

#     session.close()
#     return tokenizer


### Reference : https://medium.com/data-science-in-your-pocket/text-summarization-using-textrank-in-nlp-4bce52c5b390

## Ranking keywors for netwrok graph by keywords
@app.put("/news_summerization")
async def summerizing_news():
    session = SessionLocal()
    news = session.query(News.news_description).all()
    #dd = pd.DataFrame(news)
    #print(news)
    for abbsum in news:
        sum = abbsum#.news_description
        #print("this is sum  " + sum)
        #sum = pd.DataFrame(sum)
        #print(sum)
        sentences = sent_tokenize(str(sum))
        #print(sentences)
        vocab = {}
        preprocessed_sentences = []
        stop_words = set(stopwords.words('english')) # stopwords need more refine strings for sign 

        for sentence in sentences:
            tokenized_sentence = word_tokenize(sentence)
            result = []

            for word in tokenized_sentence:
                word = word.lower() 
                if word not in stop_words: 
                    if len(word) > 2: 
                        result.append(word)
                        if word not in vocab:
                            vocab[word] = 0
                        vocab[word] += 1
            preprocessed_sentences.append(result)
        preprocessed_sentences = preprocessed_sentences[0]  # preprocessed_sentences
        #print(preprocessed_sentences)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(preprocessed_sentences)
        encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
        max_len = max(len(item) for item in encoded)
        #padded = pad_sequences(encoded)
        #last_value = len(tokenizer.word_index) + 1
        #padded = pad_sequences(encoded, padding='post') # padded = pad_sequences(encoded, padding='post', truncating='post', maxlen=5)
        #padded = pad_sequences(encoded)#, padding='post', value=last_value)
        for sentence in encoded:
            while len(sentence) < max_len:
                sentence.append(0)

        padded_np = np.array(encoded)
        similarity_matrix = np.zeros([len(preprocessed_sentences), len(preprocessed_sentences)])

        for i,row_embedding in enumerate(padded_np):
            for j,column_embedding in enumerate(padded_np):
                similarity_matrix[i][j]=1-spatial.distance.cosine(row_embedding,column_embedding)
        #rankedkey = dict((x,y) for x,y in vocab_sorted)
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        top_sentence={sentence:scores[index] for index,sentence in enumerate(sum)}
        top=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:4])
        for sent in sentences:
            if sent in top.keys():
                print(sent)
    session.close()
    return sent

import torch
from diffusers import StableDiffusionPipeline

# Generate Journal Image 
@app.post("/gen-news-image")
def diffusion_news():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda") # gpu cuda serving only 
    session = SessionLocal()
    news = session.query(News).all()
    #damn = read_news()
    for abbsum in news:
        prompt = abbsum.news_title
        print(prompt)
        ##promptt = summerizing_news() check out this , or Do commit return value
        #prompt = # summerization string
        ##image = pipe(promptt).images[0] check out this , or Do commit return value
        image = pipe(prompt).images[0]
        img_converted = from_image_to_bytes(image)
        img_list = [img_converted, img_converted]
    return img_list # returning image type 

url = "http://localhost:8000/gen-news-image-get"

@app.get("/gen-news-image-get")
def news_gen_img():
    result = requests.get(url)
    res = json.loads(result.content)
    bytes_list = list(map(lambda x: base64.b64decode(x), res))
    image_list = list(map(lambda x: Image.open(io.BytesIO(x)), bytes_list))
    return image_list[0]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)