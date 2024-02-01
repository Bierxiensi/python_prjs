import os
from langchain_openai import OpenAI
os.environ["OPENAI_API_KEY"] = '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'

llm = OpenAI(model_name="gpt-3.5-turbo",max_tokens=200)
text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(text)