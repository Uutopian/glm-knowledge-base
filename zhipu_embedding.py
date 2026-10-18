from __future__ import annotations
import logging
from typing import Dict,List,Any
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel,root_validator
logger=logging.getLogger(__name__)

#定义一个继承Embeddings类的自定义Embeddings类
class ZhipuAIEmbeddings(BaseModel,Embeddings):

        """`Zhipuai Embeddings` embedding models."""  # docstring:类的说明书

        client: Any = None  # 字段声明:这个类有一个 client 属性(客户端)
        """`zhipuai.ZhipuAI`"""  # 这个字段的说明书(它将放 zhipuai 客户端对象)

        @root_validator(skip_on_failure=True)
        def validate_environment(cls, values: Dict) -> Dict:
            from zhipuai import ZhipuAI
            values["client"] = ZhipuAI(timeout=60, max_retries=2)
            #加固，每次调 API 最多等 30 秒，30 秒没回应就主动挂断——这就是治“卡住无限等”的那味药
            #max_retries=2	挂断后自动重试 2 次——网络抖动这种“瞬时抽风”不用你管，它自己恢复
            return values

        def _embed(self,texts:str) -> List[float]:
            embeddings=self.client.embeddings.create(
                model="embedding-2",
                input=texts
            )
            return embeddings.data[0].embedding

        def embed_documents(self,texts:List[str]) ->List[List[float]]:
            return [self._embed(text) for text in texts]

        def embed_query(self,text:str) -> List[float]:
            resp=self.embed_documents([text])
            return resp[0]
