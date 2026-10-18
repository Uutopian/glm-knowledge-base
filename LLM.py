
from zhipuai_llm import *
from dotenv import load_dotenv, find_dotenv
import os  #操作系统
import sys  #
from zhipu_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma  #向量库检索


_=load_dotenv(find_dotenv())
api_key=os.getenv("API_KEY")
sys.path.append("../FastAPIProject")
############## LangChain + 智谱 GLM 大模型 的测试代码(创建一个LLM)：##########################
zhipuai_model=ZhipuaiLLM(model_name="glm-4-flash",temperature=0,api_key=api_key)
print(zhipuai_model.invoke("你好，请你自我介绍一下！").content)


##########构建检索问答链######################
embedding=ZhipuAIEmbeddings()
#定义持久化路径，有什么用？ ----》不用每次都调用API和进行向量化
persist_directory='../vector_db/chroma'

#加载数据库
# 加载知识库文档(课程 PDF)并切片 —— from_documents 必须要文档,之前漏了这个参数
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
pages = PyPDFLoader(r"D:\FastAPIProject\knowledge_db\v1(1).pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(pages)
print(f"文档切片总数:{len(split_docs)}")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    # 库已存在:直接加载,不再调 API、不再重复入库
if vectordb._collection.count() == 0:
    vectordb = Chroma.from_documents(  #RAG的向量数据库
        documents=split_docs[:300],
        embedding=embedding,
        persist_directory=persist_directory,
    )

print(f"向量数据库存储的数量:{vectordb._collection.count()}")


print(f"向量数据库存储的数量：{vectordb._collection.count()}")

#测试加载向量数据库，使用一个问题query进行向量检索，as_retriever方法把向量数据库构造成检索器。
question='prompt'
retriever=vectordb.as_retriever(search_kwargs={"k":3})  #检索器
#docs=vectordb.similarity_search(question,k=3)
docs=retriever.invoke(question)
print(f"检索到的内容数：{len(docs)}")

for i,doc in enumerate(docs):
    print(f"检索到的第{i}个内容：\n{doc.page_content}",end="\n----------------------\n")


#构建检索链，LCEL(LangChain Expression Language, LangChain表达式语言)来构建
# workflow，LCEL可以支持异步（ainvoke）、流式(stream)、批次处理(batch)等多种运行方式，
# 同时还可以使用LangSmith无缝跟踪。
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])
#上面的combine_docs语法还不是很清楚

combiner=RunnableLambda(combine_docs)  #组合器
retrieval_chain=retriever | combiner  #数值从左向右传递，问题先被检索器得到检索结果
#再被组成器得到进一步输出

######构建检索问答链############
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
#压缩问题的系统prompt
condense_question_system_template=(
    "请根据聊天记录完善用户最新的问题，"
    "如果用户最新的问题不需要完善则返回用户的问题。"
)

#构造压缩问题的prompt template
condense_question_prompt=ChatPromptTemplate([
     ("system", condense_question_system_template),
     ("placeholder", "{chat_history}"),
     ("human", "{input}"),
])

retrieve_docs = RunnableBranch(
    # 分支 1: 若聊天记录中没有 chat_history 则直接使用用户问题查询向量数据库
    (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
    # 分支 2 : 若聊天记录中有 chat_history 则先让 llm 根据聊天记录完善问题再查询向量数据库
    condense_question_prompt | zhipuai_model | StrOutputParser() | retriever,
)

template="""使⽤以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编
造答案。最多使⽤三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {input}
"""

prompt = PromptTemplate.from_template(template)

qa_chain=(
    RunnablePassthrough.assign(context=combine_docs)
    #RunnableParallel存储用户的问题作为prompt的input
    | prompt
    | zhipuai_model
    | StrOutputParser()
)


chat_history = []
#定义历史记录的问答链
qa_history_chain=RunnablePassthrough.assign(
    context=(lambda x:x) |retrieve_docs
).assign(answer=qa_chain)




#不带聊天记录
qa_history_chain.invoke({
    "input":"西瓜书是什么？",
    "chat_history":[]
}
)

#带聊天记录
qa_history_chain.invoke({
    "input":"南瓜书跟它有什么关系？",
    "chat_history":[
        ("human","西瓜书是什么？"),
        ("ai","西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。")
    ]
})





#检测问答链效果测试
question_1="prompt有什么作用"
result=qa_history_chain.invoke({"input":question_1,"chat_history":chat_history})
print("大模型+知识库后回答 question_1 的结果：")
print(result)

print(zhipuai_model.invoke(question_1).content)




"""
#向检索链添加聊天记录
#使用 LangChain 中的ChatPromptTemplate，即将先前的对话嵌入到语言模型中
from langchain_core.prompts import ChatPromptTemplate

#问答链的系统prompt
system_prompt=(
"你是一个问答任务的助手。 "
    "请使用检索到的上下文片段回答这个问题。 "
    "如果你不知道答案就说不知道。 "
    "请使用简洁的话语回答用户。"
    "\n\n"
    "{context}"
)
#制定聊天记录
qa_prompt=ChatPromptTemplate(
    [
        ("system",system_prompt),
        ("placeholder","{chat_history}"),
        ("human","{input}")
    ]
)

#无历史记录
messages=qa_prompt.invoke(
    {
        "input":"南瓜书是什么",
        "chat_history":[],
        "context":""
    }
)
for message in messages.messages:
    print(message.content)


#有历史记录
messages=qa_prompt.invoke(
    {
        "input":""
    }
)
"""


"""
##############带有历史记录的检索链########################
chat_history = []


# 第一轮:chat_history 为空 → 自动走"直查"分支
r1 = qa_history_chain.invoke({"input":"南瓜书是什么", "chat_history": chat_history})
print("第一轮回答:", r1["answer"])

# 关键一步:把这一轮的问答追加进历史——框架不替你记,这行就是"记忆"本身
chat_history.extend([
     ("human","南瓜书是什么" ),  #历史记录要自己写
     ("ai", r1["answer"]),
])

# 第二轮:chat_history 非空 → 自动走"压缩改写"分支
# "南瓜书跟它有什么关系"里的"它",会被改写成"西瓜书"再去检索
r2 = qa_history_chain.invoke({"input": "南瓜书跟它有什么关系？", "chat_history": chat_history})
print("第二轮回答:", r2["answer"])


"""


##########总结###################
"""
1、此项目流程是搭建好环境，设置好API，巩固一些文件路径
2、搭建并加载数据库（一般会选择Chroma的向量数据库），文档进行切片并向量化加载进去
3、构建向量问答检索

"""

