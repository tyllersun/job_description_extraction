import os

import openai
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_preprocessing_func import append_dict_to_csv, process_string, read_distinct_lines
from ner_data_formating import get_ner_pair

# 从.env文件中加载环境变量
load_dotenv()

# 使用os.getenv访问你的环境变量
openai.api_key = os.getenv("openai_api_key")
openai.api_base = os.getenv("openai_api_base")
"""
job_parsed = pd.read_csv(
    "job_classification/openai_job_description_extraction/data/104.csv"
)
"""
job_parsed = read_distinct_lines("job_classification/openai_job_description_extraction/data/104.csv", n=5000)
parsed_column_name = '工作內容'

# Add response schema
response_schemas = [
    ResponseSchema(name="skill", description="list of skill"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# One shot learning, prompt
base_prompt = """
範例：Question: 需會使用Adobe In Design 或相關軟體 Answer: {"skill": ["Adobe In Design"]}
指令：把文章中提到的工作技能(skill)名稱(做NER用），按照順序放入"list of skill"中

"""

# full prompt
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="{base_prompt} \n{format_instructions} \n{question}",
    input_variables=["question"],
    partial_variables={
        "base_prompt": base_prompt,
        "format_instructions": format_instructions,
    },
)

# Create chat model
chat = ChatOpenAI(
    temperature=0, openai_api_key=openai.api_key, openai_api_base=openai.api_base
)

llm_chain = LLMChain(
    prompt=prompt,
    llm=chat,
    output_key="json_string",
)


transform_chain = TransformChain(
    input_variables=["json_string"],
    output_variables=["result"],
    transform=process_string,
)

chain = SequentialChain(
    input_variables=["question"],
    output_variables=["result"],
    chains=[llm_chain, transform_chain],
)

for i in range(len(job_parsed)):
    try:
        print(f"In job {i}, total job: {len(job_parsed)}")
        test_sentence = job_parsed[parsed_column_name][i]
        print(test_sentence)
        if str(test_sentence) == "nan":
            continue

        output = chain.run(test_sentence)

        # Append list to a text file
        with open("job_skill_ner_task/Openai_ner_task/skill_set/skill_dict.txt", "a") as f:
            for value in output["skill"]:
                f.write(value + "\t")
            f.write("\n")

        print(output)
        ner_pair = get_ner_pair(test_sentence, output["skill"])
        append_dict_to_csv(ner_pair, "job_skill_ner_task/Openai_ner_task/data/ner_data_104.csv")
    except:
        print(print(f"In job {i}, total job: {len(job_parsed)}"))
