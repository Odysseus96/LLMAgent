from vllm import LLM, SamplingParams

llm = LLM("/project/hf_model/Qwen2.5-7B-Instruct")

sampling_params = SamplingParams(temperature=0.7, top_p=0.95, repetition_penalty=1.05, max_tokens=512)

def generate_stream(prompt, history):
    full_prompt = history + '\n' + prompt

    outputs = llm.generate(full_prompt, sampling_params)
    return outputs

history = "User: What is AI?\nAssistant: AI stands for Artificial Intelligence."
for output in generate_stream("AI 能做什么？", history):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}, Generated text: {generated_text}")
