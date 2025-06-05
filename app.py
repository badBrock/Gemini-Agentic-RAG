from flask import Flask, render_template, request, jsonify
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor
from langchain.agents import load_tools
from langchain_community.document_loaders import PyPDFLoader
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
import tempfile
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, List, Any

# Custom callback handler to capture agent thinking process
class AgentThinkingCallback(BaseCallbackHandler):
    def __init__(self):
        self.thinking_process = []
        
    def on_agent_action(self, action, color=None, **kwargs):
        """Record agent actions (tool usage)"""
        self.thinking_process.append({
            'type': 'action',
            'tool': action.tool,
            'tool_input': action.tool_input,
            'log': f"Using tool: {action.tool} with input: {action.tool_input}"
        })
        
    def on_agent_finish(self, finish, **kwargs):
        """Record agent finish"""
        self.thinking_process.append({
            'type': 'finish',
            'log': f"Final Answer: {finish.return_values.get('output', '')}"
        })
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Record when LLM starts (agent thinking)"""
        # We don't want to capture all LLM calls, just high-level thinking
        pass
        
    def on_tool_end(self, output, **kwargs):
        """Record tool output"""
        self.thinking_process.append({
            'type': 'tool_output',
            'output': output,
            'log': f"Tool output: {output}"
        })

app = Flask(__name__)

# Global variables to store our RAG components
agent_executor = None
vectordb = None

def initialize_rag(pdf_path):
    """Initialize the RAG system with the given PDF file"""
    global agent_executor, vectordb
    
    # Set your API key
    os.environ["GOOGLE_API_KEY"] = "AIzaSyB3EXw0c5s6o3GlrOVB8hk_eMNWh4cT13k"
    
    # Load and split documents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Create vector store
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.from_documents(docs, embedding)
    retriever = vectordb.as_retriever()
    
    # Setup RAG QA chain
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    # Create wrapper function for QA tool
    def qa_tool_wrapper(query):
        result = qa_chain({"query": query})
        return result["result"]
    
    # Define routing helper functions
    def is_definition_query(query):
        """Check if the query is asking for a definition."""
        patterns = [
            r'(define|definition of|meaning of|what (is|are)|who (is|are)|explain the term)\s+\w+',
            r'what does .+ mean',
        ]
        return any(re.search(pattern, query.lower()) for pattern in patterns)

    def is_math_query(query):
        """Check if the query is a math calculation."""
        patterns = [
            r'\d+\s*[\+\-\*\/\^\(\)]\s*\d+',
            r'(calculate|compute|solve|evaluate)\s+.+',
            r'(sum|product|difference|quotient|square root|log|sin|cos|tan)\s+of',
            r'what is\s+\d+\s*[\+\-\*\/\^]\s*\d+'
        ]
        return any(re.search(pattern, query.lower()) for pattern in patterns)

    def is_document_query(query):
        """Check if the query is about the document."""
        patterns = [
            r'(document|text|paragraph|chapter|story|theme|character|plot|narrative|book|article|passage)',
            r'(in|from|about)\s+the\s+(document|text|story|book|passage)',
            r'what (is|are|does) .+ (in|say|about|mention)'
        ]
        return any(re.search(pattern, query.lower()) for pattern in patterns)

    # Tool selection router
    def router_function(query):
        if is_definition_query(query):
            return "This is asking for a definition. I should use the ddg-search tool to look up the meaning."
        elif is_math_query(query):
            return "This appears to be a calculation. I should use the llm-math tool to compute the result."
        elif is_document_query(query):
            return "This is a question about the document content. I should use the RAGQA tool."
        else:
            return "I need to analyze this query further to determine the best tool."
    
    # Create the tools
    tool_list = load_tools(["llm-math", "ddg-search"], llm=llm)
    
    # Add document QA tool
    tool_list.append(
        Tool(
            name="RAGQA",
            func=qa_tool_wrapper,
            description="Useful for answering questions about the content of the document."
        )
    )
    
    # Add router tool
    router_tool = Tool(
        name="QueryRouter",
        func=router_function,
        description="ALWAYS use this first to determine which other tool you should use for a given query."
    )
    
    # Add the router tool to the beginning of the list
    tool_list.insert(0, router_tool)
    
    # Create a custom prompt that encourages proper tool usage
    custom_prompt_template = """You are an AI assistant with access to several tools:

1. QueryRouter: ALWAYS use this tool first to decide which other tool you should use next
2. ddg-search: Use this ONLY for definitions and general information lookup
3. llm-math: Use this ONLY for mathematical calculations
4. RAGQA: Use this ONLY for questions about the document content

Follow these rules strictly:
- For definition questions like "Define X" or "What is X", use ddg-search
- For calculation questions like "Calculate X" or math problems, use llm-math
- For document-related questions, use RAGQA
- ALWAYS start with QueryRouter to help you decide

Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always start by using the QueryRouter tool
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""
    
    # Create the custom prompt template
    custom_prompt = PromptTemplate.from_template(custom_prompt_template)
    
    # Create the agent with the custom prompt
    agent = create_react_agent(
        llm=llm,
        tools=tool_list,
        prompt=custom_prompt
    )
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tool_list,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=True  # Return intermediate steps for debugging
    )
    
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        # Save the uploaded file to a temporary location
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)
        
        # Initialize RAG with the uploaded PDF
        success = initialize_rag(temp_file_path)
        
        if success:
            return jsonify({'success': 'RAG system initialized successfully with ' + file.filename})
        else:
            return jsonify({'error': 'Failed to initialize RAG system'}), 500
    else:
        return jsonify({'error': 'Only PDF files are supported'}), 400

@app.route('/query', methods=['POST'])
def process_query():
    global agent_executor
    
    if agent_executor is None:
        return jsonify({'error': 'Please upload a PDF file first'}), 400
    
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Process the query using the agent with verbose=True to capture the agent's thinking
        response = agent_executor.invoke(
            {"input": query},
            {"callbacks": [AgentThinkingCallback()]}  # Custom callback to capture thinking
        )
        
        # Get the thinking process from our custom callback
        thinking = getattr(response, "thinking_process", [])
        
        # Extract intermediate steps from response if available
        intermediate_steps = response.get("intermediate_steps", [])
        
        # Format intermediate steps into a more readable structure
        thinking_process = []
        for step in intermediate_steps:
            # First element is the action
            action = step[0]
            # Second element is the observation (result)
            observation = step[1]
            
            thinking_process.append({
                'type': 'action',
                'tool': action.tool,
                'tool_input': action.tool_input,
                'log': f"Using tool: {action.tool} with input: {action.tool_input}"
            })
            
            thinking_process.append({
                'type': 'tool_output',
                'output': observation,
                'log': f"Tool output: {observation}"
            })
        
        # Return the response with the thinking process
        return jsonify({
            'answer': response["output"],
            'thinking': thinking_process,
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

if __name__ == '__main__':
    app.run(debug=True)