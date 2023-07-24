# use OpenAI GPT to extract information from job description
import openai
import pandas as pd
from func import append_dict_to_csv, process_string
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate

openai.api_key = "XXXXXX"
openai.api_base = "XXXXXXX"


# 設定回傳格式
response_schemas = [
    ResponseSchema(
        name="company_description", description="infomation about the company"
    ),
    ResponseSchema(name="people_need", description="number of people want to hire"),
    ResponseSchema(name="job_duty", description="工作要做的事情"),
    ResponseSchema(name="job_requirement", description="需要的技能和基本要求"),
    ResponseSchema(name="skill_nice_to_have", description="非必要技能，但有會更好"),
    ResponseSchema(name="welfare", description="公司福利"),
    ResponseSchema(name="other", description="非以上的部分"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# 設定prompt模板
prompt = PromptTemplate(
    template="把段句話，分到對應的類別，並整理成Json，缺少欄位填NULL  \n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

chat = ChatOpenAI(
    temperature=0, openai_api_key=openai.api_key, openai_api_base=openai.api_base
)

# chain 架構
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

# 輸入要做parsing的資料
data = pd.read_csv("data/meet_job_2023-06-29.csv")


list_of_parse_dict = []
column_name_list = list(data.columns)
filename = "./data/job_parsed.csv"
for i in range(len(data["jd"])):
    parse_dict = {}
    for column in column_name_list:
        parse_dict[column] = data[column][i]
    try:
        parsed_data = chain.run(question=data["jd"][i])
        parse_dict["company_description"] = parsed_data["company_description"]
        parse_dict["people_need"] = parsed_data["people_need"]
        parse_dict["job_duty"] = parsed_data["job_duty"]
        parse_dict["job_requirement"] = parsed_data["job_requirement"]
        parse_dict["skill_nice_to_have"] = parsed_data["skill_nice_to_have"]
        parse_dict["welfare"] = parsed_data["welfare"]
        parse_dict["other"] = parsed_data["other"]
    except Exception:
        parse_dict["company_description"] = ""
        parse_dict["people_need"] = ""
        parse_dict["job_duty"] = ""
        parse_dict["job_requirement"] = ""
        parse_dict["welfare"] = ""
        parse_dict["other"] = ""
    append_dict_to_csv([parse_dict], filename)
    list_of_parse_dict.append(parse_dict)


print(list_of_parse_dict)
