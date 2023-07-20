from dotenv import load_dotenv
import os
from langchain.chains import LLMChain, TransformChain
from langchain.chains import SequentialChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import openai
import pandas as pd
from langchain_preprocessing_func import process_string, append_dict_to_csv
from ner_data_formating import get_ner_pair

# 从.env文件中加载环境变量
load_dotenv()

# 使用os.getenv访问你的环境变量
openai.api_key  = os.getenv('openai_api_key')
openai.api_base = os.getenv('openai_api_base')

job_parsed = pd.read_csv("openai_job_description_extraction/data/meet_job_2023-06-29.csv")

# Add response schema
response_schemas = [
    ResponseSchema(name="skill", description="list of skill"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# One shot learning, prompt
base_prompt = """
把文章中提到的技能(skill)名稱(做NER用），按照順序放入"list of skill"中
例如：Familiar in server side languages (C# .Net)則要取出：C#, .Net
"""

# full prompt
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="{base_prompt} \n{format_instructions} \n{question}",
    input_variables=["question"],
    partial_variables={"base_prompt":base_prompt, "format_instructions": format_instructions}
)

# Create chat model
chat = ChatOpenAI(
            temperature=0,
            openai_api_key=openai.api_key,
            openai_api_base=openai.api_base
    )

llm_chain = LLMChain(
    prompt=prompt,
    llm=chat,
    output_key="json_string",
)


transform_chain = TransformChain(
    input_variables=["json_string"],
    output_variables=["result"],
    transform=process_string
)

chain = SequentialChain(
    input_variables=["question"],
    output_variables=["result"],
    chains=[llm_chain, transform_chain],
)

for i in range(len(job_parsed)):
    print(f"In job {i}, total job: {len(job_parsed)}")
    test_sentence = job_parsed['jd'][i]
    if str(test_sentence) == "nan":
        continue

    output = chain.run(test_sentence)
    print(output)
    ner_pair = get_ner_pair(test_sentence, output["skill"])
    append_dict_to_csv(ner_pair, 'Openai_ner_task/data/ner_data.csv')

