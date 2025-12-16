import ollama

response = ollama.generate(
    model='deepseek-r1:1.5b',
    prompt='Say hello in one sentence'
)

print(response['response'])
