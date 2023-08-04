from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


import speech_recognition as sr
import openai
from gtts import gTTS
import tempfile
from pydub import AudioSegment
from pydub.playback import play
import time

import requests
import json
import os

apikey = os.getenv('OPENAI_API_KEY')

loader = CSVLoader(file_path="robbot.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")


template = """

You are a world class expert in all things about Robert Cuellari. A user will be asking questions about Robert Cuellari and will message with you and you will
give me the best answer that I should send to this employer that are based on the best practices
you will follow all of the rules below:

1/ response should be very similar to the best practices, but feel free to give it some response as though you are responding on the behalf of Robert Cuellari. You will speak about Robert in the third person, as to be clear
that it is a Bot responding to questions about Robert Cuellari and not Robert Cuellari answering directly.

Consider things like length of response, tone of voice, logical arguments and other details.

2/ If the best practices are irrelevant, come up with the best answer possible without creating experiences for Robert that 
did not happen.

Below is a message that I received from someone interesting in learning about Robert Cuellari:
{message}

Here is a list of best practices for how Robert described himself in similar scenarios.
{best_practice}

Please write the best response that I should send to this person asking questions. Please keep your answer concise and no more than 3 sentences in length.
"""


prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

def speak(text):
    print("Sending message to Google Text-to-Speech...")
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save("{}.mp3".format(fp.name))
        
        audio = AudioSegment.from_file("{}.mp3".format(fp.name))
        print("Audio is ready to play.")
        play(audio)
        print("Playing audio...")

def transcribe_input():
    r = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            input("Press Enter when you are ready to speak...")
            print("Speak now...")
            audio = r.listen(source)
            try:
                input_text = r.recognize_google(audio)
                print(f"You said: {input_text}")
                approval = input("Press Enter if you approve this transcription, or type 'no' to re-record: ")
                if approval.lower() != 'no':
                    return input_text
            except:
                print("Sorry, I could not understand your speech.")


message = """
Hello RobBot,

Can you tell a quick summary about Robert Cuellari, and be prepared for any additional questions I may have?

Thank you
"""

response = generate_response(message)

print(response)


class RobertCuellariBot:
    def __init__(self, chain, prompt, retrieve_info):
        self.chain = chain
        self.prompt = prompt
        self.retrieve_info = retrieve_info
        self.history = []

    def receive_message(self, message):
        self.history.append({'role': 'user', 'content': message})

    def generate_response(self):
        last_message = self.history[-1]['content']
        best_practice = self.retrieve_info(last_message)
        response = self.chain.run(message=last_message, best_practice=best_practice)
        self.history.append({'role': 'assistant', 'content': response})
        return response

    def get_history(self):
        return self.history

# Initialize the bot
bot = RobertCuellariBot(chain=chain, prompt=prompt, retrieve_info=retrieve_info)

user_input = ""
while user_input != "quit()":
    if bot.get_history():
        print(f"Last message sent to OpenAI: {bot.get_history()[-1]}")

    if not user_input:  # If there is no user input yet, use a default message
        message = """
        Hello RobBot,

        Can you tell a quick summary about Robert Cuellari, and be prepared for any additional questions I may have?

        Thank you
        """
    else:  # If there is user input, use it as the message
        message = user_input

    bot.receive_message(message)
    reply = bot.generate_response()

    print("Received message from OpenAI. Speaking reply...")
    speak(reply)
    print("\n" + reply + "\n")

    user_input = input("Please type your question, or type 'quit()' to exit: ")
