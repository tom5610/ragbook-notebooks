import boto3

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_aws import ChatBedrock

from anthropic import AnthropicBedrock

bedrock_runtime = boto3.client(service_name="bedrock-runtime")

BEDROCK_HAIKU_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

def completion(
        prompt_template: PromptTemplate,
        parameters: dict = {},
        bedrock_model_id:str = BEDROCK_HAIKU_MODEL_ID, 
        max_tokens:int = 2048,
        temperature:float = 1,
        top_k:int = 50,
        top_p:float = 0.5,
        output_parser:BaseOutputParser = StrOutputParser()) -> str:

    model_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }
    model = ChatBedrock(
        client=bedrock_runtime,
        model_id=bedrock_model_id,
        model_kwargs=model_kwargs
    )

    chain = prompt_template | model | output_parser
    response = chain.invoke({**parameters})
    return response


client = AnthropicBedrock(aws_region="us-east-1")

MODEL_NAME = "anthropic.claude-3-haiku-20240307-v1:0"

def __get_completion(prompt, system=""):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2048,
        temperature=0.0,
        messages=[
            {"role": "user", "content": prompt}
        ],
        system=system
    )
    return message.content[0].text

