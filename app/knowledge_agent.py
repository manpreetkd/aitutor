import easyocr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from crewai import Agent, Task, Crew, Process, LLM
from langchain_ollama import OllamaLLM
from knowledge_rag import KnowledgeRAG
import uuid
import os

ALLOWED_IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
ALLOWED_DOC_EXTENSIONS = {'.pdf', '.txt'}

class TutorAgent:
    def __init__(self):
        self.rag = KnowledgeRAG()
        self.vectorstore_path = "vectorstore" 
        self.reader = easyocr.Reader(['en'])  # OCR for text extraction
        self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def extract_text_from_image(self, image_path):
        """Extract text from an image using EasyOCR."""
        extracted_text = self.reader.readtext(image_path, detail=0)
        return " ".join(extracted_text) if extracted_text else None  # Return None instead of an empty string

    def describe_image(self, image_path):
        """Generate a caption for an image using BLIP model."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.vision_processor(image, return_tensors="pt")
        caption_ids = self.vision_model.generate(**inputs)
        return self.vision_processor.decode(caption_ids[0], skip_special_tokens=True)

    def get_and_store_documents(self, uploaded_files, image_file=None):
        file_names, img_file_names = [], []

        for file in uploaded_files + (image_file or []):
            file_ext = os.path.splitext(file.name)[1].lower()
            
            if file_ext in ALLOWED_DOC_EXTENSIONS:
            # Store PDFs/documents in knowledge folder
                save_path = os.path.join('knowledge/', file.name)
                with open(save_path, 'wb') as input_files:
                    input_files.write(file.getbuffer())
                file_names.append(file.name)
            
            elif file_ext in ALLOWED_IMG_EXTENSIONS:
            # Store images in root folder
                save_path = file.name
                with open(save_path, 'wb') as input_files:
                    input_files.write(file.getbuffer())
                img_file_names.append(file.name)

        return file_names, img_file_names
    
    def process_documents(self, uploaded_files, input_question, image_upload=None):
        file_names, img_file_names = self.get_and_store_documents(uploaded_files, image_upload)
        OllamaLLM(model="ollama/llama3.2", temperature=0.2)

        all_content = ""
        extracted_image_text = None
        image_description = None

        vectorstore_path = f"vectorstore_{uuid.uuid4()}"  

        if file_names:
            for pdf_path in file_names:
                file_path = os.path.join('knowledge/', pdf_path)
                self.rag.setup_knowledge_base(file_path, vectorstore_path) 

            retrieved_context = self.rag.get_relevant_context(input_question, vectorstore_path)
            if retrieved_context:
                all_content += retrieved_context

        # Process Images If Uploaded
        if img_file_names:
            image_path = img_file_names[0]
            extracted_image_text = self.extract_text_from_image(image_path)
            image_description = self.describe_image(image_path)

        # Build Context Dynamically
        combined_context = f"{all_content}\n\n"
        if extracted_image_text:
            combined_context += f"Extracted Image Text: {extracted_image_text}\n"
        if image_description:
            combined_context += f"Image Description: {image_description}\n"

        # Initialize Tutor Agent
        tutor_agent = Agent(
            role="Encouraging & Motivational Tutor",
            goal="Empower students with positive reinforcement and insightful guidance.",
            backstory="You are a patient and supportive tutor who helps students learn by encouraging their curiosity and critical thinking skills.",
            llm=LLM(model="ollama/llama3.2", temperature=0.5),
            verbose=True
        )

        # Define Task to Ensure Answering Only from the Document
        task = Task(
            description=f"""
            You are a supportive tutor dedicated to motivating students while providing clear, concise answers. Follow the following steps to generate the response. Go through the following steps to provide the final response.
            
            Step 1: Start with Encouragement
            - Always begin with an encouraging statement, for example: 
              - "That's a great question! I love your curiosity."
              - "You're thinking in the right direction—let's explore this together!"
              - "Fantastic inquiry! Let's break it down step by step."

            Step 2: Provide a Structured Answer
            - Use the provided context below to answer the student's question.
            - If the question relates to a document, answer using the document knowledge.
            - If an image is uploaded, analyze the extracted content only if relevant.

            Context from Documents & Images:
            {combined_context}

            Question: {input_question}

            Step 3: Encourage Further Learning
            - End with a motivational closing, e.g.,
              - "You're doing great! Keep exploring new ideas."
              - "Curiosity is the key to learning—keep asking great questions!"
              - "Fantastic effort! Let’s keep building on this knowledge."

            Step 4: Accuracy & Relevance
            - If the answer is in the provided documents, base your response on them.
            - If the image contains useful content, integrate it.
            - If the relevant answer is not found, DO NOT MAKE UP INFORMATION. Instead, say:
              - "I'm not entirely sure about this, but I’d love to help you explore it further!"

            """,
            expected_output="An encouraging, structured, and motivational response with document-based knowledge. Starting should be an encouraging statement and ending should be a motivational closing.",
            agent=tutor_agent
        )

        # Run CrewAI Process
        crew = Crew(
            agents=[tutor_agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential,
            embedder={"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
        )

        inputs = {"question": input_question, "context": combined_context}
        outcome = crew.kickoff(inputs=inputs)

        return outcome.raw
