# RAG 
# using the model: zai-glm-4.7

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know.
"""

# user prompt has placeholders for the question and context 
USER_PROMPT_TEMPLATE = """

Question: 
{question}

Context: 
{context}

"""


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="zai-glm-4.7"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model


    def search(self, question, num_results= 5):
        
        return self.index.search(
            question,
            num_results = num_results
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["content"])
            lines.append(doc["filename"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, question, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=question, context=context
        )
    
    def llm(self, prompt):
        input_messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.chat.completions.create(
                model="zai-glm-4.7",
                messages= input_messages
        )
            
        return response
    

    def rag(self, question):
        search_results = self.search(question)
        prompt = self.build_prompt(question, search_results)
        response = self.llm(prompt)           
        answer = response.choices[0].message.content
        usage = response.usage.prompt_tokens # usage = input token
    
        return answer, usage
    













