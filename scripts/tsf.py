from openai import OpenAI

client = OpenAI(
    api_key="sk-ftylzstkkpxtxvwribwvcbsadlfuadgayetxfuvlnbsrcikj",
    base_url="https://api.siliconflow.cn/v1",
)
model_name = "Qwen/Qwen3.5-397B-A17B"
model_name = "Qwen/Qwen3.5-4B"
model_name = "Pro/deepseek-ai/DeepSeek-V3.2"
# model_name = "Qwen/Qwen3.5-27B"
# 发送带有流式输出的请求
system_text = 'You are an expert in motion captioning for CLIP embedding. Given two independent descriptions from a motion capture dataset:\r\n \
- Upper body style: "{style_description}"\r\n \
- Lower body movement: "{movement_type}"\r\n \
Create a single, concise, natural English sentence (maximum 30 words) that describes the complete full-body human motion. \
Ensure the sentence is coherent, flows logically, and is optimized for CLIP text encoding: use vivid yet professional action verbs, \
avoid redundancy, and start with an action-oriented structure. Do not add any extra explanation or conflict resolution unless necessary. \
Output only the final sentence.\r\n \
Prioritize natural language suitable for CLIP: begin with "A person" or "Human motion of", \
use precise biomechanics terms if applicable, and ensure semantic independence is resolved into a unified action.'

user_input = 'style_description="Both arms raised as wings – tilting from side to side" movement_type="Backwards Running"'

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": f"{system_text}"},
        {"role": "user", "content": f"{user_input}"},
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=False,
)
print("Response:")
content = response.choices[0].message.content
print("模型回复：")
print(content)

# print(response)
# 逐步接收并处理响应
# for chunk in response:
#     if not chunk.choices:
#         continue
#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="", flush=True)
#     if chunk.choices[0].delta.reasoning_content:
#         print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
# print("")
