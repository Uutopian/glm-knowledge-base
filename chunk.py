import os
from dotenv import load_dotenv,find_dotenv
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders import PyPDFLoader
#构建Chroma向量库
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter



#读取本地的环境变量,加载API路径
find_dotenv()  #寻找并定位.env文件的路径
load_dotenv()  #读取该.env文件
_=load_dotenv(find_dotenv())

#定义embeddings
embedding=ZhipuAIEmbeddings()



#获取folder_path下所有文件路径，存储在file_paths里
file_paths=[]
folder_path='.'


for root,dirs,files in os.walk(folder_path):  #遍历文件夹，root根目录(路径)，dirs子目录
    for file in files:
        file_path=os.path.join(root,file)  #这里是什么意思
        file_paths.append(file_path)

#遍历文件路径，并把实例化的loader存入loaders里
loaders=[]  #应该是文件路径，转换为list类型，那之前的文件不是list类型吗（file_paths)
for file_path in file_paths:
    file_type=file_path.split('.')[-1]#先判断属于pdf还是mdS
    if file_type == 'pdf':
        loaders.append(PyPDFLoader(file_path))
    if file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))  #创建加载器实例，记录文件路径

texts=[]
for loader in loaders:
    texts.extend(loader.load())  #每一项拆开，逐个放进原列表，把文件变成程序看得懂的Document


#########################组文档切分####################################
pattern=re.compile(r'[\u4e00-\u9fff](\n)[^\u4e00-\u9fff]',re.DOTALL)
for text in texts:
    text.page_content=re.sub(  #查找加替换
        pattern,
        lambda m:m.group(0).replace(m.group(0),''),
        text.page_content   #原始文件

    )

#先创建分割器（参数卸载构造里
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n","\n","。","？","，",""],

)
print(splitter.split_text(text.page_content[0:1000]))  #先取前1000看效果
split_docs=splitter.split_documents(texts)


#定义持久化路径，有什么用？ ----》不用每次都调用API和进行向量化
persist_directory='./vector_db/chroma'
vectordb=Chroma.from_documents(
    documents=split_docs[:20],
    embedding=embedding,
    persist_directory=persist_directory
)
vectordb.persist()
print(f"向量数据库中储存的数量：{vectordb._collection.count()}")


#以上的路径是  文件路径加载-------》把文件变为程序可读的document-----》文切分----》存入向量数据库中


#向量检索
question="什么是大语言模型"
"""



sim_docs=vectordb.similarity_search(question,k=3)  #chroma采用的是余弦相似度
print(len(sim_docs))
for i,sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
"""
#避免结果过于单一，采用最大边际相关性MMR
mmr_docs=vectordb.max_marginal_relevance_search(question,k=3)
for i,sim_doc in enumerate(mmr_docs):
    print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

