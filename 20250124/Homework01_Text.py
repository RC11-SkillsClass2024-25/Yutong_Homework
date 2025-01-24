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




####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################



def search_paragraphs(query_text, n):
    import gensim
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from langdetect import detect  # 用于检查语言
    from ebooklib import epub
    from bs4 import BeautifulSoup
    
    # 全局列表，用于存储所有书籍的处理结果
    all_processed_docs = []
    
    # 读取 epub 文件中的段落
    def read_epub_paragraphs(epub_file, ID):
        # 读取 epub 文件
        book = epub.read_epub(epub_file)
        paragraphs = []
        
        # 遍历章节，提取文本
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.content, 'html.parser')
                for para in soup.find_all('p'):
                    paragraphs.append(para.get_text())
        
        return paragraphs
    
    # 判断段落是否是英语文本
    def is_english_text(text):
        try:
            # 使用 langdetect 来检测语言
            return detect(text) == 'en'
        except:
            return False
    
    # 处理每本书的文本
    def readingprocess(epub_file, ID):
        paragraphs = read_epub_paragraphs(epub_file, ID)
        
        processed_docs = []  # 临时存储每本书的处理结果
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        STOP_WORDS = stopwords.words("english")
        
        for i, paragraph in enumerate(paragraphs):
            if is_english_text(paragraph):  # 确保段落是英文
                # 分词处理
                words = gensim.utils.simple_preprocess(paragraph, min_len=3, deacc=True)
                
                # 词形还原
                lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
                
                # 去除停用词
                filtered_words = [word for word in lemmatized_words if word not in STOP_WORDS]
                
                # 词干提取（可选择是否需要）
                stemmed_words = [stemmer.stem(word) for word in filtered_words]
                
                # 确保处理后的文档不为空
                if len(stemmed_words) > 0:
                    # 拼接为一个字符串
                    processed_doc = " ".join(stemmed_words)
                    # 将处理后的文档加入临时列表
                    #processed_docs.append({'processed_doc': processed_doc, 'bookID': ID})
                    processed_docs.append({'processed_doc': processed_doc, 'nr': i, 'bookID': ID})
                else:
                    # 如果该段落处理后没有有效单词，跳过或打印警告
                    print(f"Warning: Paragraph from Book ID {ID} resulted in an empty processed document.")
        
        # 返回处理后的文档
        return processed_docs
        


    
    # 调用处理函数并将结果添加到全局列表
    def process_books():
        # 依次处理每本书
        all_processed_docs.extend(readingprocess('epubs/The Art-Architecture Complex - Hal Foster.epub', 1))
        all_processed_docs.extend(readingprocess('epubs/About Looking - John Berger.epub', 2))
        all_processed_docs.extend(readingprocess('epubs/Blowup - Michelle Kasprzak ed.epub', 3))
    
    # 执行处理
    process_books()

    
    #矢量化
    from sklearn.feature_extraction.text import TfidfVectorizer #形成tfidf矩阵
    texts = [doc['processed_doc'] for doc in all_processed_docs]
    vectorizer = TfidfVectorizer(min_df=2) #形成tfidf矩阵
    tfidf_matrix = vectorizer.fit_transform(texts) #形成tfidf矩阵
    
    #导入相似度
    from sklearn.metrics.pairwise import cosine_similarity #导入相似度
    
    #降维
    import numpy as np
    from scipy.sparse import random
    from sklearn.decomposition import TruncatedSVD # also known as Latent Semantic Analysis (LSA)
    n_components = 100 #独特词减少到100个，即保留最重要的100个独特词
    svd = TruncatedSVD(n_components=n_components, algorithm = 'randomized') #这是计算 SVD 分解的更快、随机的方法
    reduced_matrix = svd.fit_transform(tfidf_matrix) #得到一个形状为(n_samples, n_components)的矩阵，其中n_samples是段落数量，n_components（独特词）数量为 100
    #前两行的作用是制定降维的规则
    
    #处理&调用查询词
    def preprocess_query(query_text):
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        STOP_WORDS = stopwords.words("english")

        # 分词
        words = gensim.utils.simple_preprocess(query_text, min_len=3, deacc=True)

        # 词形还原
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

        # 去除停用词
        filtered_words = [word for word in lemmatized_words if word not in STOP_WORDS]

        # 词干提取（可选择是否需要）
        stemmed_words = [stemmer.stem(word) for word in filtered_words]

        # 返回处理后的文本
        return " ".join(stemmed_words)
        
    processedQuery = preprocess_query(query_text) #把词语按照处理书籍的方式还原和筛选词汇
    query_vector = vectorizer.transform([processedQuery]) #将处理过的词汇转为向量
    query_vector.toarray().flatten().argsort()[::-1]
    #对文档或查询中的术语的重要性进行排序。
    #获取向量（例如 TF-IDF 向量）中最重要的术语的索引。
    #确定查询中具有最高值的顶级特征（单词）
    reduced_query_vec = svd.transform(query_vector) #将之前的向量转为降维的向量
    similarities2 = cosine_similarity(reduced_query_vec, reduced_matrix) #查询降维后向量与每一个段落（降维后矩阵中的每一行）之间的相似度\
    
    top_n = int(n)
    top_n_indices = similarities2[0].argsort()[::-1][:top_n]  # 获取前四个最相似文档的索引
    
    # 打印前四个最相似文档的索引和相似度得分
    print(f"Top {top_n} most similar documents to the query:")
    for i in top_n_indices:
        print(f"Document Index: {i}")
        print(f"Document: {all_processed_docs[i]['processed_doc']}")
        print(f"Document: {all_processed_docs[i]['bookID']}")
        print(f"Document: {all_processed_docs[i]['nr']}")
        print()
    



####################################################################################################################################################################
####################################################################################################################################################################
#################################################################### LLM Method ！！！##########################################################
def search_paragraph(query_text):
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
    ####################################################################################################################
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
    ####################################################################################################################
    
    
       
    
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
    
    
    ####################################################################### LLM Model ##############################################
    from FlagEmbedding import FlagModel
    
    model = FlagModel('BAAI/bge-large-en-v1.5',
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:")
    embeddings1 = model.encode([p['paragraph'] for p in all_paragraphs]) #通过模型 model 将一组段落文本转换为语义嵌入（embeddings）。
    queries = model.encode_queries([query_text]) #输入查询词
    similarities = embeddings1 @ queries.T #计算所有向量与查询词的相关性
    top_index = similarities.argmax() #找到最高相关性的向量
    
    return all_paragraphs[top_index] #最高相关性向量 句子