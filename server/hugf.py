import requests
from huggingface_hub import InferenceClient,ChatCompletionOutput
from huggingface_hub import login
# Your Hugging Face API token

login(token="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")
def hugf_Model(prompt):
    API_TOKEN = ""
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    client = InferenceClient(
        provider="nebius",
        model="Qwen/QwQ-32B",
        api_key=API_TOKEN,
    )
    return client.chat_completion(messages, max_tokens=1000)['choices'][0]['message']['content']


# import requests
# from huggingface_hub import InferenceClient,ChatCompletionOutput
# # Your Hugging Face API token

# def hugf_Model(prompt):
#     API_TOKEN = "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"
#     messages = [
#         {
#             "role": "user",
#             "content": prompt,
#         }
#     ]

#     client = InferenceClient(
#         provider="together",
#         model="meta-llama/Meta-Llama-3-8B-Instruct",
#         api_key=API_TOKEN,
#     )
#     return client.chat_completion(messages, max_tokens=1000)['choices'][0]['message']['content']


