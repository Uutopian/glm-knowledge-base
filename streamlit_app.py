#作用在应用上写代码，

import streamlit as st
from zhipuai import ZhipuAI
import json
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch,RunnablePassthrough
import sys

from zhipu_embedding import ZhipuAIEmbeddings
from zhipuai_llm import *
from langchain_community.vectorstores.chroma import Chroma  #向量库检索
from dotenv import load_dotenv, find_dotenv

_=load_dotenv(find_dotenv())
api_key=os.getenv("API_KEY")
sys.path.append("../FastAPIProject")
##########定义函数,返回一个检索器##########
# ========== 历史对话持久化:每轮对话写进 json,启动时读回 ==========

HISTORY_FILE = 'chat_history.json'

def load_history():
    """启动时从 json 文件读取历史对话(跨刷新/跨重启记忆)"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []          # 文件损坏时宁可空开始,不让应用崩
    return []

def save_history(messages):
    """每轮对话后把历史写回 json 文件"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def get_retriever():
    embedding=ZhipuAIEmbeddings()

    #向量库持久化路径(建议改成绝对路径,避免随运行目录漂移)
    persist_directory='./vector_db/chroma'
    #加载已有的向量数据库(只加载,不重建、不重复入库)
    vectordb=Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()

#该函数处理检索器返回的文本
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

############返回一个检索问答链#########
def get_qa_history_chain():
    retriever=get_retriever()
    llm=ZhipuaiLLM(model_name="glm-4-flash",temperature=0,api_key=api_key) #不传入api_key,从环境变量读取
    condense_question_system_template=(
        "请根据聊天记录总结用户最近的问题,"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    # 注意:placeholder 的第二个元素必须是字符串 "{chat_history}"
    condense_question_prompt=ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    retrieve_docs=RunnableBranch(
        # 分支 1: 没有聊天记录 → 直接用用户问题查询向量库
        (lambda x:not x.get("chat_history",False),(lambda x:x["input"])|retriever,),
        # 分支 2: 有聊天记录 → 先让 llm 根据聊天记录完善问题再查询
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt=(
        "你是一个问答任务的助手。"
        "请使用检索到的上下文片段回答这个问题。"
        "如果你不知道答案就说不知道。"
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )

    qa_prompt=ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    qa_chain=(
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # 官方对话式 RAG 的组装方式:先注入 context(检索结果),再基于 context 生成 answer
    # 注意:这里用 RunnablePassthrough.assign,不是 RunnableBranch(它不支持关键字参数)
    qa_history_chain=RunnablePassthrough.assign(
        context=retrieve_docs
    ).assign(answer=qa_chain)

    return qa_history_chain


##########接收检索问答链,用户输入及聊天历史,以流式返回##########
def get_response(chain,input,chat_history):
    response=chain.stream({
        "input": input,
        "chat_history": chat_history
    })

    for res in response:
        if "answer" in res.keys():
            yield res["answer"]


#定义main函数,制定显示效果与逻辑
def main():
    st.markdown('### 🦜🔗 动手学大模型应用开发 · 个人知识库助手')
    # st.session_state 存储用户与应用交互期间的状态与数据
    # 注意:必须是 "not in" —— 不存在才初始化;原来写成 in 会把历史反复清空
    if "messages" not in st.session_state:
        st.session_state.messages=load_history()
    # 存储检索问答链(只构建一次)
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain=get_qa_history_chain()
    # 建立容器,高度为500px
    messages=st.container(height=500)
    # 展示整个对话历史
    for message in st.session_state.messages:
        with messages.chat_message(message[0]): #message[0] 是角色(human/ai)
            st.write(message[1])  #message[1] 是内容

    if prompt := st.chat_input("Say something"):
        st.session_state.messages.append(("human",prompt))
        save_history(st.session_state.messages)

        with messages.chat_message("human"):
            st.write(prompt)

        answer=get_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )

        with messages.chat_message("ai"):
            # 最稳方案:先把流式结果拼成完整文本,再一次性写入
            # (放弃打字机动画,换取零前端冲突;升级 Streamlit 后可恢复打字机)
            full_text="".join(answer)
            st.markdown(full_text)

        st.session_state.messages.append(("ai",full_text))
        save_history(st.session_state.messages)


if __name__ == "__main__":
    main()



##################

#1、为什么这个文件没有文档切片，把文档将加入向量数据库中