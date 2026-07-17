import os
import tempfile
import gradio as gr
from google import genai
from gtts import gTTS


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def extract_text(file):
    if file is None:
        return "Please upload a file first.", None

    uploaded_file = client.files.upload(file=file.name)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            uploaded_file,
            "Extract all text exactly as written."
        ]
    )

    return response.text, None



def summarize(file):
    if file is None:
        return "Please upload a file first.", None

    uploaded_file = client.files.upload(file=file.name)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            uploaded_file,
            "Summarize the document in simple language."
        ]
    )

    return response.text, None

def answer_question(file, question):
    if file is None:
        return "Please upload a file first.", None

    if not question.strip():
        return "Please enter a question.", None

    uploaded_file = client.files.upload(file=file.name)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            uploaded_file,
            f"Answer the following question based on the uploaded document:\n{question}"
        ]
    )

    return response.text, None



def audio_summary(file):
    if file is None:
        return "Please upload a file first.", None

    uploaded_file = client.files.upload(file=file.name)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            uploaded_file,
            "Summarize the document in simple language."
        ]
    )

    summary = response.text

    audio_path = tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False
    ).name

    gTTS(text=summary, lang="en").save(audio_path)

    return summary, audio_path


with gr.Blocks(title="Gemini Document Analyzer") as demo:

    gr.Markdown(
        """
        # 📄 Gemini Document Analyzer

        Upload a PDF or Image and choose one of the following:
        - Extract Text
        - Summarize
        - Audio Summary
        - Ask Questions
        """
    )

    with gr.Row():

        with gr.Column(scale=1):
            file = gr.File(
                label="Upload PDF/Image"
            )

        with gr.Column(scale=2):

            gr.Markdown("## Choose an Action")

            with gr.Row():
                btn_extract = gr.Button("Extract Text")
                btn_summary = gr.Button("Summarize")
                btn_audio = gr.Button("Audio Summary")

            gr.Markdown("---")

            question = gr.Textbox(
                label="Ask a Question",
                placeholder="Example: What is the main topic?"
            )

            btn_qa = gr.Button(
                "Answer Question",
                variant="primary"
            )

    gr.Markdown("---")

    output = gr.Textbox(
        label="Output",
        lines=15
    )

    audio = gr.Audio(
        label="Audio Summary"
    )

    btn_extract.click(
        fn=extract_text,
        inputs=file,
        outputs=[output, audio]
    )

    btn_summary.click(
        fn=summarize,
        inputs=file,
        outputs=[output, audio]
    )

    btn_audio.click(
        fn=audio_summary,
        inputs=file,
        outputs=[output, audio]
    )

    btn_qa.click(
        fn=answer_question,
        inputs=[file, question],
        outputs=[output, audio]
    )


if __name__ == "__main__":
    demo.launch()
