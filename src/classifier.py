import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field


def gen_system_prompt():
    prompt = """
    You are an assistant with knowledge and expertise in political science, social sciences, rule of law, and related fields. 
    Your task is to assist me in classifying news articles in two groups: those related to Judicial Independence and those unrelated to it.
    To successfully accomplish this task, you will have to carefully read a summary of a news article that I will provide. Then, based in 
    the definition of Judicial Independence that I will give you and your own knowledge, you will assess if the news article is related or 
    unrelated to my definition of Judicial Independence.
    """

    return prompt


def gen_instructions_prompt(article):
    prompt = f"""
    To help you contextualize, this is the definition of Judicial Independence that we are using for the task at hand:

    Judicial independence is the concept that the judiciary should be independent from the other branches of government, by means of having 
    sufficient resources and professional judges, with adequate rights and competencies, as well as its ability to impose disciplinary 
    measures on government officials. Allowing judges to make decisions impartially, without fear of punishment or expectation of reward, 
    and free of corruption and undue influence from political and private interests.

    Now, given the following news article:
    {article}
    [END OF ARTICLE]

    Please classify the article as either related or unrelated to Judicial Independence. If you consider that the news article is related,
    answer TRUE. If you consider that the news article is unrelated, answer FALSE. Please answer with a single word: TRUE or FALSE.
    Do NOT add any additional comments or explanations to your answer.

    Provide an answer following a JSON format as follows:
    {{
        "status": "<your_answer>"
    }}

    Thank you.
    """

    return prompt


class Status(BaseModel):
    status: str = Field(description="Answer to the classification task, either TRUE or FALSE.")


def classify_article(api_key, article):
    """
    Classify a news article as related or unrelated to Judicial Independence.

    Args:
        article (str): The summary of the news article to classify.

    Returns:
        str: The classification result, either "TRUE" or "FALSE".
    """

    openai_client = OpenAI(api_key = api_key)

    history = [
        {"role": "system", "content": gen_system_prompt()},
        {"role": "user",   "content": gen_instructions_prompt(article)}
    ]

    completion = openai_client.beta.chat.completions.parse(
        messages = history,
        model    = "gpt-4o-2024-08-06",
        response_format = Status
    )

    result = completion.choices[0].message.parsed
    
    return result.status


def process_data(df, excluded_articles, api_key):
    """
    Process the DataFrame to classify articles.

    Args:
        api_key (str): The OpenAI API key.
        df (pd.DataFrame): The DataFrame containing the articles to classify.

    Returns:
        pd.DataFrame: The DataFrame with the classification results.
    """

    reduced_data = df[~df["id"].isin(excluded_articles)]
    data = reduced_data.copy()
    data["token_count"] = data["summary"].apply(lambda x: len(x.split()))
    data["cum_token_count"] = data["token_count"].cumsum()

    batch = (
        data.copy()
        .loc[data["cum_token_count"] <= 500000]
    )
    batch["status"] = batch["summary"].apply(lambda x: classify_article(api_key, x))
    
    return batch    