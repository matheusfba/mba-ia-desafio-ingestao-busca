import google.generativeai as genai

genai.configure(api_key="AIzaSyD-jXUP3BLfCCnm8Ha5x6Cn9LgD1XLPH5o")

for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(m.name)