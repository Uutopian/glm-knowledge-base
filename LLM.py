
from zhipuai_llm import *  #导入智谱大语言模型相关包
from dotenv import load_dotenv, find_dotenv  #加载文件.env的环境变量
import os  #操作系统接口模块，用来获取环境变量
import sys  #系统模块，添加模块搜索路径
from zhipu_embedding import ZhipuAIEmbeddings   #使用封装好的词向量化
from langchain_community.vectorstores.chroma import Chroma  #向量库检索


_=load_dotenv(find_dotenv())  #获取.env路径，load_dotenv读取.env，find_dotenv自动寻找.env的文件路径
api_key=os.getenv("API_KEY")  #为什么要这样写，前面不是已经获取路径了吗，从环境变量中找到API key
#这里load_dotenv(find_dotenv)将.env加载到环境路径，没有真正读取具体值
sys.path.append("../FastAPIProject") #加载文件路径，为什么？因为要添加pytohn的模块搜索路径，便于zhipuai_llm找到路径
############## LangChain + 智谱 GLM 大模型 的测试代码(创建一个LLM)：##########################
zhipuai_model=ZhipuaiLLM(model_name="glm-4-flash",temperature=0,api_key=api_key) #加载api_key和大模型，温度0，确定性高
print(zhipuai_model.invoke("你好，请你自我介绍一下！").content)  #这个语法是什么.invoke和content
#.invoke是执行这个链/模型；LLM返回AIMessage对象(**消息对象**，存回答文本、工具调用、token 消耗等信)
#。content提取其中的文本内容，也就是返回的问答

##########构建检索问答链######################
embedding=ZhipuAIEmbeddings()  #词向量化？
#定义持久化路径，有什么用？ ----》不用每次都调用API和进行向量化，为什么不用
persist_directory='./vector_db/chroma'  #建立向量数据库

#加载数据库
# 加载知识库文档(课程 PDF)并切片 —— from_documents 必须要文档,之前漏了这个参数
from langchain_community.document_loaders import PyPDFLoader   #PDF切分
from langchain_text_splitters import RecursiveCharacterTextSplitter  #？传入这个包有什么用，做数据切分
# 自动加载 knowledge_db 文件夹里的所有 PDF(以后加文档:丢进文件夹重跑即可)
import glob #？有什么作用，加载pdf吗
pages = []  #记录pdf的页数？为后面切分文档时提供资源？
for f in glob.glob(r"D:\FastAPIProject\knowledge_db\*.pdf"):
    pages += PyPDFLoader(f).load()  #返回一个列表，每个元素都是document对象
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#进行数据切分，切分的大小每次有500个字符，两个词块间有50个字符重叠，防止一句话中断，这里数据还没转化为向量吗---》是的，这里只是长文档切成小块
split_docs = splitter.split_documents(pages)  #把切分好的字符转换成文件赋值到split_docs中
print(f"文档切片总数:{len(split_docs)}")  #查看文档切片数，后续存放到向量数据库中
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)  #为什么要与这一步--》这里是加载已有的向量数据库
if vectordb._collection.count() == 0:    #为什么要加上这个if语句-》上面那行那个代码只是加载数据库，这里指的是如果数据库是空的，就创建新的
    vectordb = Chroma.from_documents(  #RAG的向量数据库
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_directory,  #持久化路径，不用每次都进行向量化
    )

print(f"向量数据库存储的数量:{vectordb._collection.count()}")


#########以上是数据向量化过程#########################################

############进行验证，判断向量数据库是否良好？##############################
#测试加载向量数据库，使用一个问题query进行向量检索，as_retriever方法把向量数据库(vectordb)构造成检索器。
question='prompt'
retriever=vectordb.as_retriever(search_kwargs={"k":3})  #检索器,每次返回3个检索结果
#docs=vectordb.similarity_search(question,k=3)
docs=retriever.invoke(question)  #检索关于问题的内容，retriever是检索器对象，而.invoke()是执行检索
print(f"检索到的内容数：{len(docs)}")

for i,doc in enumerate(docs):  #enumerate是内置函数，给列表添加索引
    print(f"检索到的第{i}个内容：\n{doc.page_content}",end="\n----------------------\n")


#构建检索链，LCEL(LangChain Expression Language, LangChain表达式语言)来构建
# workflow，LCEL可以支持异步（ainvoke）、流式(stream)、批次处理(batch)等多种运行方式，
# 同时还可以使用LangSmith无缝跟踪。  LangSmith是什么-》LangChain官方的调试平台，可追踪链的执行过程
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  #检索问答链


#这个函数是什么意思-》文档合并，多个文档片段合并成一个上下文
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"]) #把生成的文本内容合并成一个上下文
#上面的combine_docs语法还不是很清楚

combiner=RunnableLambda(combine_docs)  #组合器，为什么需要组合器
#这里combine_docs不是LangChain对象，而RunnableLambda把普通函数包装成LangChain可运行的对象，才能用“|”连接
retrieval_chain=retriever | combiner  #数值从左向右传递，问题先被检索器得到检索结果
#这里
#“|”是管道符，A|B其本质是：创建一个链，先执行A，把结果传给B

######构建检索问答链############       （提示词模板）   （聊天格式的提示词模板）
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate  #这个是提示词工程的内容吗
from langchain_core.output_parsers import StrOutputParser  #这个有什么作用-》把LLM输出转化成字符串，如果没有则LLM会返回AI message对象（列表）
from langchain_core.runnables import RunnableBranch   #这个又是什么作用-》条件分支，根据条件走不同路径
#压缩问题的系统prompt，我觉得RAG优化和这个也有关系
condense_question_system_template=(
    "请根据聊天记录完善用户最新的问题，"
    "如果用户最新的问题不需要完善则返回用户的问题。"
)

#构造压缩问题的prompt template，为什么，有什么作用-》把多轮对话的模糊问题改写成独立完整的问题
condense_question_prompt=ChatPromptTemplate([
     ("system", condense_question_system_template),
     ("placeholder", "{chat_history}"),  #这个有什么用-》placeholder是占位符，存放聊天记录
     ("human", "{input}"),  #用户输入的prompt
])

retrieve_docs = RunnableBranch(   #问题压缩，在多轮对话中补全问题
    # 分支 1: 若聊天记录中没有 chat_history 则直接使用用户问题查询向量数据库
    (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
    # 分支 2 : 若聊天记录中有 chat_history 则先让 llm 根据聊天记录完善问题再查询向量数据库
    condense_question_prompt | zhipuai_model | StrOutputParser() | retriever,
)

#这个也是优化问答质量的吗-》是的，这是控制LLM输出质量的关键
template="""使⽤以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编
造答案。最多使⽤三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {input}
"""

#提示词
prompt = PromptTemplate.from_template(template)

#检索问答链，问答链是什么意思-》把多个步骤串起来，用管道符
qa_chain=(                   #（将新字段填充到输入字典里）
    RunnablePassthrough.assign(context=combine_docs)  #这个语法是什么-》合并文档，这里是决定combine_docs中的docs['context']里context的命名
    #RunnableParallel存储用户的问题作为prompt的input
    | prompt  #填充提示词
    | zhipuai_model  #经过智谱大模型
    | StrOutputParser()  #转字符串
)


chat_history = []  #保存历史纪录
#定义历史记录的问答链，为什么要这样写-》完整的问答链，整合了检索+QA
qa_history_chain=RunnablePassthrough.assign(
    context=(lambda x:x) |retrieve_docs  #检索文档
).assign(answer=qa_chain)  #用检索结果生成文档




#不带聊天记录，测试
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


设置一个RAG系统，解决三个核心问题：
   1、知识怎么存  -》文档-切片-向量化-加入向量数据库
   2、问题怎么查 -》用户提问-检索相关文档
   3、答案怎么生 ->文档+问题--喂给LLM-生成答案
"""

