import os

import ebooklib
from ebooklib import epub
import re
import os

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words, stopwords, names

ENGLISH_WORDS = set(words.words())

import gensim

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from langdetect import detect  # 用于检查语言
from bs4 import BeautifulSoup


###################################################################################################
###################################################################################################  LLM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def merge_strings_until_limit(strings, min_length, max_length, test_for_max = 0):
        merged_string = ""
        merged_strings = []
        
        for s in strings:
            if len(merged_string) <= min_length:
                merged_string += s
            
            elif len(merged_string) > max_length and test_for_max<5:
                    splitParagraph = merged_string.split('.')
                    splitParagraphRePoint = []
                    for sp in splitParagraph:
                        splitParagraphRePoint.append(sp+'.')
                    
                    merged = merge_strings_until_limit(splitParagraphRePoint, min_length, max_length, test_for_max+1)
                    merged_strings.extend(merged)
                    merged_string = s
            else:
                merged_strings.append(merged_string)
                merged_string = s
        
        if merged_string:
            merged_strings.append(merged_string)
        
        return merged_strings

#####################################################################################################
all_paragraphs = []
def read_epub_paragraphs(epub_file, ID):
    book = epub.read_epub(epub_file)
    paragraphs = []
    
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content().decode('utf-8')
        content = re.sub('<[^<]+?>', '', content)
        content = re.sub('\s+', ' ', content)
        content = re.sub('\n', ' ', content)
        
        paragraphs.extend(content.strip().split("&#13;"))
    
    paragraphs = merge_strings_until_limit(paragraphs, 200, 1000)
    paragraphs = [{'paragraph':paragraphs[i], 'nr':i, 'bookID':ID} for i in range(len(paragraphs))]
    
    return paragraphs[1:-1]



def process_books():
    # 依次处理每本书
    all_paragraphs.extend(read_epub_paragraphs('epubs/The Art-Architecture Complex - Hal Foster.epub', 1))
    all_paragraphs.extend(read_epub_paragraphs('epubs/About Looking - John Berger.epub', 2))
    all_paragraphs.extend(read_epub_paragraphs('epubs/Blowup - Michelle Kasprzak ed.epub', 3))

process_books()

#####################################################################################################
def search_paragraph(query_text):
    from FlagEmbedding import FlagModel
    
    model = FlagModel('BAAI/bge-large-en-v1.5',
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:")
    embeddings1 = model.encode([p['paragraph'] for p in all_paragraphs]) #通过模型 model 将一组段落文本转换为语义嵌入（embeddings）。
    queries = model.encode_queries([query_text]) #输入查询词
    similarities = embeddings1 @ queries.T #计算所有向量与查询词的相关性
    top_index = similarities.argmax() #找到最高相关性的向量
    
    return all_paragraphs[top_index] #最高相关性向量 句子





###################################################################################################
###################################################################################################  TFIDF!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

