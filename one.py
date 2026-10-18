from langchain_community.document_loaders import PyMuPDFLoader
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader=PyMuPDFLoader(r"D:\小鹅通课件\MySQL课件\MySQL课件\1.初识MySQL\课程内容\初识MySQL.pdf")
pattern=re.compile(r'[\u4e00-\u9fff](\n)[^\u4e00-\u9fff]',re.DOTALL)
pdf_pages=loader.load()
pdf_page=pdf_pages[1]
pdf_page.page_content=re.sub(pattern,lambda match:match.group(0).replace('\n','').replace('\r',''),pdf_page.page_content)

#先创建分割器（参数卸载构造里
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n","\n","。","？","，",""],

)
splitter.split_text(pdf_page.page_content[0:1000])  #先取前1000看效果
split_docs=splitter.split_documents(pdf_pages)
print(f"切分后的文件数量：{len(split_docs)}")
print(f"切分后的字符数（可用来大致评估token数）：{sum([len(doc.page_content) for doc in split_docs])}")
#再对文本调用（只能传str)
chunks=splitter.split_text(pdf_page.page_content)
"""
#re.sub(正则规则，替换逻辑，原始文本）
pdf_page.page_content=pdf_page.page_content.replace('.','')
pdf_page.page_content=pdf_page.page_content.replace(' ','')
#结构.replace('old','new')
print(pdf_page.page_content)
"""

"""
print(f"{type(pdf_pages)}",f"{len(pdf_pages)}")
pdf_page=pdf_pages[1]
print(f"每一个元素的类型{type(pdf_page)}",
      f"该文档的描述性数据:{pdf_page.metadata}",
      f"查看该文档的内容:\n{pdf_page.page_content}",
      sep="\n---------\n"
      )

"""


#搭建并使用向量数据库
