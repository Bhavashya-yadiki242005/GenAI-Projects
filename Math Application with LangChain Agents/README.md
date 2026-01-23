Math Application with LangChain Agents (OCR + LLM)

This project is a Math Solver Web Application built using LangChain Agents, OCR, and LLMs.
It allows users to upload an image containing a math problem (printed or handwritten), automatically extracts the text, and then solves the problem step by step, finally showing the answer.

Instead of manually typing questions from textbooks or worksheets, this app reads directly from images and performs reasoning using a Large Language Model.

Key Features

Upload math questions as images

OCR-based text extraction using Tesseract

Step-by-step math reasoning using LLMs

LangChain-powered prompt handling

Simple and interactive Gradio web UI


Tech Stack Used
Component	Technology
OCR	pytesseract
Image Processing	Pillow (PIL)
LLM Reasoning	OpenAI GPT (via LangChain)
Agent Framework	LangChain
UI Framework	Gradio
Language	Python
Step 1: Install Dependencies

Install all required Python libraries using the command below:

pip install langchain langchain-community langchain-openai openai duckduckgo-search ddgs pytesseract Pillow gradio


Important:
Make sure Tesseract OCR is installed on your system:

Windows:
Download from → https://github.com/UB-Mannheim/tesseract/wiki

Linux:

sudo apt install tesseract-ocr


Mac:

brew install tesseract

Step 2: Environment Setup (API Key)

Set your OpenAI API key as an environment variable.

Option 1: Inside Python (for testing)
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

Option 2: System Environment Variable (Recommended)

Windows (PowerShell):

setx OPENAI_API_KEY "your-api-key"


Linux / Mac:

export OPENAI_API_KEY="your-api-key"


You may also replace OpenAI with Gemini or other supported LLMs.

Step 3: Import Libraries (Explanation of Each)
import gradio as gr
from PIL import Image
import pytesseract
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate

Explanation:

gradio → Builds the web interface

PIL.Image → Loads and processes images

pytesseract → Extracts text from images (OCR)

ChatOpenAI → Connects to OpenAI LLM

LangChain Tools & Agents → Manages AI logic

PromptTemplate → Formats structured prompts

Step 4: Optical Character Recognition (OCR)
Cell Explanation
def extract_text(image):
    return pytesseract.image_to_string(image)

What this does:

Takes an image as input

Extracts readable text using OCR

Returns raw text from the image

Step 5: LLM Setup & Math Prompt
llm = ChatOpenAI(model="gpt-4", temperature=0)

Explanation:

Uses GPT-4 for accurate math reasoning

temperature=0 ensures deterministic output

Prompt Template
math_prompt = PromptTemplate(
    input_variables=["question"],
    template="Solve this math related reasoning step by step and give only the final answer: {question}"
)

Purpose:

Forces the model to reason step-by-step

Ensures a clean final answer

Math Solver Function
def solve_math(query):
    return llm.invoke(math_prompt.format(question=query)).content

What happens:

Sends extracted text to the LLM

Returns the final computed answer

Step 6: Processing Pipeline (Core Logic)
def process_image(image):
    text = extract_text(image)
    if not text.strip():
        return "No text detected in the image."
    answer = solve_math(text)
    return f"Extracted Question:\n{text}\n\n Final Answer: {answer}"

Flow Explanation:

Receives image from user

Extracts text using OCR

Checks if text exists

Sends question to LLM

Returns both question + answer

Step 7: Gradio Web Interface
with gr.Blocks(css="footer {display:none !important;}") as demo:


Creates the UI container and hides Gradio footer.

UI Layout
gr.Markdown("# Math Solver with LangChain and OCR")


Displays project title.

img_input = gr.Image(type="pil", label=" Upload Your Question Image")
output_box = gr.Textbox(label="Result", lines=10)


Image upload input

Text output box

Event Handling
img_input.upload(process_image, img_input, output_box)


Triggers processing when image is uploaded

clear_btn.click(lambda: "", None, output_box)


Clears output on button click

Step 8: Run the Application
demo.launch(share=True)

Output:

Launches a local web app

Generates a public Gradio link

Allows image upload via:

File upload

Webcam

Clipboard paste

Sample Output

Upload a math question image

Extracted text is displayed

Final answer is shown clearly

Use Cases

Students solving textbook problems

Teachers checking answers

Exam preparation

Accessibility for handwritten notes

AI-based education tools

Future Enhancements

Show step-by-step reasoning

Advanced math tools (symbolic math)

Voice input support

Mobile-friendly UI

Agent-based tool routing

Conclusion

This project demonstrates how LangChain Agents + OCR + LLMs can be combined to build a real-world AI application.
It showcases skills in AI reasoning, multimodal input handling, and web deployment, making it ideal for final-year projects and resumes.
