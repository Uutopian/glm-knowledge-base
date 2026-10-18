from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

loader=UnstructuredMarkdownLoader(r"D:\typora笔记内容\操作系统原理.md")
md_pages=loader.load()
print(f"蔡汝侯变量的类型为：{type(md_pages)}",f"改Markdown一共包含{len(md_pages)}")
md_page=md_pages[0]
print(f"么一个元素的类型：{type(md_page)}",
      f"该文档的描述性数据：{md_page.metadata}",
      f"查看该文档的内容:\n{md_page.page_content[0:][:200]}",
      sep="\n---------\n"
      )
