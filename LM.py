from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re



def lm_model(prompt):
  # Tokenize the input prompt

  model_id = "microsoft/phi-1_5"  # or another chat model

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, )

  #prompt = "Tell me a story about a prince."

  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")

  # Generate the text
  generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=300,
    use_cache=True,
  )
  out=tokenizer.decode(generation_output[0])
  out=re.split(r'Exercise 2:', out)[0]
  out=re.split(r'(2)', out)[0]
  out = out.replace(prompt, "", 1)
  return out
  # Print the output
  #print(out)

# chat = pipeline("text-generation", model=model, tokenizer=tokenizer,)

# prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."

# response = chat(prompt, max_new_tokens=250, do_sample=False, use_cache=True )
# print(response[0]['generated_text'])
