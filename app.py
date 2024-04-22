import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import pipeline

model_name_or_path = "TheBloke/meditron-70B-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)
system_message = "This is a system message."

prompt = "What is the role of AI in managing cardiovascular diseases?"
prompt_template = f'''system
{system_message}
user
{prompt}
assistant
'''
tokens = tokenizer(
    prompt_template,
    return_tensors='pt'
).input_ids.cuda()

generation_params = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1
}

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_output_streamed = model.generate(
    tokens,
    streamer=streamer,
    **generation_params
)
token_output_streamed = generation_output_streamed[0]
text_output_streamed = tokenizer.decode(token_output_streamed)

st.write("model.generate(streamed) output: ", text_output_streamed)


@st.cache(allow_output_mutation=True)
def generate_text(prompt_template):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **generation_params
    )
    pipe_output = pipe(prompt_template)[0]["generated_text"]
    return pipe_output


pipe_output = generate_text(prompt_template)
st.write(pipe_output)