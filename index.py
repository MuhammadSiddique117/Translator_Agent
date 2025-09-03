from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI-API-KEY")
# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

transtalor = Agent(
    name = "Translator Agent",
    instructions= "You can translate Paragraph,Essay,Poems,Idoms etc in different languages. translate urdu into english "
)


response = Runner.run_sync(
    transtalor,
    input = """
ایجنٹک اے آئی ایسی مصنوعی ذہانت ہے جو خودمختار طور پر فیصلے کرنے اور کام انجام دینے کی صلاحیت رکھتی ہے۔
یہ نظام صرف انسان کے احکامات پر عمل نہیں کرتا بلکہ اپنے اہداف خود متعین کر سکتا ہے،
حالات کا تجزیہ کر کے مؤثر اقدامات کر سکتا ہے، اور اپنے فیصلوں کے نتائج سے سیکھنے کی صلاحیت بھی رکھتا ہے۔ 
ایجنٹک اے آئی میں خود مختار سیکھنے، مقصد پر مبنی رویہ، اور ماحول کے مطابق ردِ عمل جیسی خصوصیات شامل ہوتی ہیں۔
اس ٹیکنالوجی کے استعمال سے مختلف شعبوں جیسے صحت، تعلیم، اور روبوٹکس میں بہتری آ سکتی ہے،
لیکن ساتھ ہی اس کے اخلاقی، سماجی اور سیکیورٹی سے متعلق خدشات بھی پیدا ہوتے ہیں۔
""",
    run_config = config
    )
print(response.final_output)