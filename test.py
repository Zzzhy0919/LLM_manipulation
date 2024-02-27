import openai

openai.api_base = "https://ai98.vip/v1"
openai.api_key = "sk-CcCvez7msME97Kwp5cDb5947E8D7466a94Cf400b21285d0d"

chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": "ping!"}]
)

print(chat_completion.choices[0].message.content)