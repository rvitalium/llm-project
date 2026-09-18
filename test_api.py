from google import genai

client = genai.Client()

response = client.models.generate_content(model="gemini-3.8-flash", contents="What is 2+2?")

print(response.text)