import openai
import os
import time
# This code is for v1 of the openai package: pypi.org/project/openai
from openai import OpenAI

client = OpenAI(api_key="your api key")


def gpt_call(prompt="How are you?"):
    time.sleep(2)
    test_chat_message = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",  # "gpt-3.5-turbo-1106",
        messages=test_chat_message,
        temperature=0.7,
        max_tokens=2048,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response)

    return response.choices[0].message.content  # response



