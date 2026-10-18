
import os
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv
# 读取本地/项⽬的环境变量。
# find_dotenv() 寻找并定位 .env ⽂件的路径
# load_dotenv() 读取该 .env ⽂件，并将其中的环境变量加载到当前的运⾏环境中
# 如果你设置的是全局的环境变量，这⾏代码则没有任何作⽤。



_=load_dotenv(find_dotenv())








client=ZhipuAI(
    api_key=os.environ["ZHIPUAI_API_KEY"]
)   #api客户端




def gen_glm_params(prompt):
    messages=[{"role":"user","content":prompt}]
    return messages

def get_completion(prompt,model="glm-4-flash",top_p=0.7):
    messages=gen_glm_params(prompt)  #叫用接口
    response=client.chat.completions.create(
        model=model,
        messages=messages,
        top_p=top_p,
    )
    if len(response.choices)>0:
        return response.choices[0].message.content
    return "generate answer error"


from zhipuai import ZhipuAI
def zhipu_embedding(text:str):
    api_key=os.environ["ZHIPUAI_API_KEY"]
    client=ZhipuAI(api_key=api_key)
    response=client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return response

text='要生成embedding的输入文本，字符串形式'
response=zhipu_embedding(text=text)
print(f'response类型为{type(response)}')
print(f'embedding的类型为{response.object}')
print(f'生成embedding的长度为{len(response.data[0].embedding)}')
print(f'embedding(前10)为:{response.data[0].embedding[:10]}')
