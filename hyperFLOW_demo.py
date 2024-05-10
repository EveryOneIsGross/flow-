from typing import Any, Callable, Dict, List, Optional, Union
from tiktoken import get_encoding
from bs4 import BeautifulSoup
from gpt4all import Embed4All
from textblob import TextBlob
from datetime import datetime
from openai import OpenAI
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from io import StringIO
import numpy as np
import requests
import hashlib
import sqlite3
import json
import uuid
import spacy
from spacy.tokens import Span
import pickle
import PyPDF2
import os
import re
import csv

# spacy models needed
# ``python -m spacy download en_core_web_trf``

class Resources:
    def __init__(self, resource_type: str, resource_path: str, context_template: str = None):
        self.resource_type = resource_type
        self.resource_path = resource_path
        self.context_template = context_template
        if resource_type != 'sql':
            self.data = self.load_resource()
        self.chunks = []

    def load_resource(self):
        if self.resource_type == 'text':
            return self.load_text()
        elif self.resource_type == 'pdf':
            return self.load_pdf()
        elif self.resource_type == 'web':
            return self.load_web()
        elif self.resource_type == 'sql':
            return self.load_sql()
        else:
            raise ValueError(f"Unsupported resource type: {self.resource_type}")

    def load_sql(self):
        if os.path.exists(self.resource_path):
            return self.resource_path
        else:
            return None

    def load_text(self):
        with open(self.resource_path, 'r') as file:
            return file.read()

    def load_pdf(self):
        with open(self.resource_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text

    def load_web(self):
        response = requests.get(self.resource_path)
        return response.text

    def chunk_resource(self, chunk_size: int, overlap: int = 0):
        chunker = TextChunker(self.data, chunk_size, overlap)
        self.chunks = chunker.chunk_text()

    def contextualize_chunk(self, chunk: Dict[str, Any]) -> str:
        if self.context_template:
            return self.context_template.format(
                chunk=chunk['text'],
                file=self.resource_path,
                start=chunk['start'],
                end=chunk['end']
            )
        else:
            return chunk['text']

# TOOLS

class TextChunker:
    def __init__(self, text: str = None, chunk_size: int = 1000, overlap: int = 0):
        self.text = text
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = get_encoding("cl100k_base")

    def chunk_text(self, text: str = None, chunk_size: int = None, start_pos: int = 0) -> List[Dict[str, Any]]:
        if text is not None:
            self.text = text
        if chunk_size is not None:
            self.chunk_size = chunk_size

        tokens = self.encoding.encode(self.text)
        num_tokens = len(tokens)

        chunks = []
        current_pos = start_pos

        while current_pos < num_tokens:
            chunk_start = max(0, current_pos - self.overlap)
            chunk_end = min(current_pos + self.chunk_size, num_tokens)

            chunk_tokens = tokens[chunk_start:chunk_end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append({
                "text": chunk_text,
                "start": chunk_start,
                "end": chunk_end
            })

            current_pos += self.chunk_size - self.overlap

        return chunks

class TextCleaner:
    def __init__(self, text: str):
        self.text = text

    def clean_text(self) -> str:
        cleaned_text = self.text
        cleaned_text = self.remove_special_characters(cleaned_text)
        cleaned_text = self.remove_extra_whitespace(cleaned_text)
        return cleaned_text

    def parse_table_content(self, table: str) -> str:
        output = StringIO()
        writer = csv.writer(output)
        for row in table.strip().split('\n'):
            columns = re.split(r'\t|,', row.strip())
            writer.writerow(columns)
        return output.getvalue()

    def remove_special_characters(self, text: str) -> str:
        return re.sub(r'[^\w\s,]', '', text)

    def remove_extra_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()


class TextReaderTool:
    def __init__(self, resource: Resources, chunk_size: int, num_chunks: int):
        self.resource = resource
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks

    def read_text(self) -> List[Dict[str, Any]]:
        self.resource.chunk_resource(self.chunk_size)
        contextualized_chunks = [
            {
                'text': self.resource.contextualize_chunk(chunk),
                'start': chunk['start'],
                'end': chunk['end'],
                'file': self.resource.resource_path
            }
            for chunk in self.resource.chunks[:self.num_chunks]
        ]
        return contextualized_chunks

class WebScraperTool:
    def __init__(self, resource: Resources, chunk_size: int, num_chunks: int):
        self.resource = resource
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks

    def scrape_text(self) -> List[Dict[str, Any]]:
        # Load the HTML content from the resource
        html_content = self.resource.load_resource()
        # Create a BeautifulSoup object to parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style tags
        for tag in soup(['script', 'style']):
            tag.decompose()
        # Extract the text content from the HTML
        text = soup.get_text(separator='\n')
        # Split the text into chunks
        chunks = self.split_text_into_chunks(text)
        # Contextualize the chunks
        contextualized_chunks = [
            {
                'text': self.resource.contextualize_chunk(chunk),
                'start': chunk['start'],
                'end': chunk['end'],
                'file': self.resource.resource_path
            }
            for chunk in chunks[:self.num_chunks]
        ]
        return contextualized_chunks

    def split_text_into_chunks(self, text: str) -> List[Dict[str, Any]]:
        # Split the text into chunks based on the specified chunk size
        chunks = []
        start = 0
        end = self.chunk_size
        while start < len(text):
            # Find the nearest line break or end of text
            newline_pos = text.find('\n', end)
            if newline_pos == -1:
                end = len(text)
            else:
                end = newline_pos
            chunk = {
                'text': text[start:end].strip(),
                'start': start,
                'end': end
            }
            chunks.append(chunk)
            start = end + 1
            end = start + self.chunk_size
        return chunks

class NERExtractionTool:
    def __init__(self, text: str = None):
        self.text = text
        self.nlp = spacy.load("en_core_web_trf")

    def extract_entities(self, text: Optional[str] = None) -> List[Dict[str, Any]]:
        if text is not None:
            self.text = text
        doc = self.nlp(self.text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })

        return entities

class SemanticAnalysisTool:
    def __init__(self, text: str = None):
        self.text = text

    def analyze_sentiment(self, text: Optional[str] = None) -> Dict[str, Any]:
        if text is not None:
            self.text = text
        blob = TextBlob(self.text)
        sentiment = blob.sentiment
        return {
            "polarity": sentiment.polarity,
            "subjectivity": sentiment.subjectivity
        }

class KnowledgeGraphTool:
    def __init__(self, input_data: Union[str, Dict, List], input_format: str = "text"):
        self.input_data = input_data
        self.input_format = input_format
        self.nlp = spacy.load("en_core_web_trf")  # Using a more advanced NER model

    def parse_input(self):
        if self.input_format == "text":
            return self.input_data
        elif self.input_format == "json":
            # Parse JSON input and extract relevant text
            pass
        elif self.input_format == "csv":
            # Parse CSV input and extract relevant text
            pass
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")

    def extract_entities(self):
        text = self.parse_input()
        doc = self.nlp(text)
        entities = [{"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in doc.ents]
        return entities

    def extract_relationships(self, entities):
        # Implement advanced relationship extraction techniques
        pass

    def construct_graph(self, entities, relationships, output_format: str = "mermaid"):
        if output_format == "mermaid":
            # Generate Mermaid syntax for the knowledge graph
            pass
        elif output_format == "json":
            # Generate JSON representation of the knowledge graph
            pass
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def update_graph(self, new_entities, new_relationships):
        # Implement incremental graph update logic
        pass

class KnowledgeGraphTool:
    def __init__(self, input_data: Union[str, Dict, List], input_format: str = "text"):
        self.input_data = input_data
        self.input_format = input_format
        self.nlp = spacy.load("en_core_web_trf")  # Using a more advanced NER model

    def parse_input(self):
        if self.input_format == "text":
            return self.input_data
        elif self.input_format == "json":
            # Parse JSON input and extract relevant text
            if isinstance(self.input_data, dict):
                return self.extract_text_from_json(self.input_data)
            elif isinstance(self.input_data, list):
                return "\n".join([self.extract_text_from_json(item) for item in self.input_data])
            else:
                raise ValueError("Invalid JSON input format")
        elif self.input_format == "csv":
            # Parse CSV input and extract relevant text
            if isinstance(self.input_data, list):
                return "\n".join([",".join(row) for row in self.input_data])
            else:
                raise ValueError("Invalid CSV input format")
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")

    def extract_text_from_json(self, json_data: Dict) -> str:
        text = ""
        for key, value in json_data.items():
            if isinstance(value, str):
                text += f"{key}: {value}\n"
            elif isinstance(value, dict):
                text += f"{key}:\n{self.extract_text_from_json(value)}\n"
            elif isinstance(value, list):
                text += f"{key}:\n" + "\n".join([self.extract_text_from_json(item) for item in value if isinstance(item, dict)])
        return text.strip()

    def extract_entities(self):
        text = self.parse_input()
        doc = self.nlp(text)
        entities = [{"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in doc.ents]
        return entities

    def extract_relationships(self, entities):
        relationships = []
        for ent1 in entities:
            for ent2 in entities:
                if ent1 != ent2:
                    relationship = self.find_relationship(ent1, ent2)
                    if relationship:
                        relationships.append((ent1["text"], relationship, ent2["text"]))
        return relationships

    def find_relationship(self, ent1: Dict, ent2: Dict) -> Optional[str]:
        # basic implementation of relationship extraction based on entity labels
        if ent1["label"] == "PERSON" and ent2["label"] == "ORG":
            return "works_for"
        elif ent1["label"] == "PERSON" and ent2["label"] == "PERSON":
            return "knows"
        elif ent1["label"] == "ORG" and ent2["label"] == "GPE":
            return "located_in"
        else:
            return None

    def construct_graph(self, entities, relationships, output_format: str = "mermaid"):
        if not entities:
            summary = self.generate_summary()
            if output_format == "mermaid":
                return f"No entities found to construct a knowledge graph. Summary: {summary}"
            elif output_format == "json":
                return json.dumps({"summary": summary}, indent=2)

        if output_format == "mermaid":
            mermaid_syntax = "graph TD;\n"
            for entity in entities:
                mermaid_syntax += f'    {entity["text"].replace(" ", "_")}("{entity["text"]}");\n'
            for rel in relationships:
                mermaid_syntax += f'    {rel[0].replace(" ", "_")} -->|"{rel[1]}"| {rel[2].replace(" ", "_")};\n'
            return mermaid_syntax
        elif output_format == "json":
            graph = {
                "entities": entities,
                "relationships": [
                    {"source": rel[0], "target": rel[2], "type": rel[1]}
                    for rel in relationships
                ]
            }
            return json.dumps(graph, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def update_graph(self, new_entities, new_relationships):
        # Implement incremental graph update logic
        pass

    def generate_summary(self):
        text = self.parse_input()
        doc = self.nlp(text)
        sentences = [sent.text.capitalize() for sent in doc.sents]
        summary = " ".join(sentences[:3])  # Use the first three sentences as the summary
        return summary

class UserFeedbackTool:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def request_feedback(self, context: str) -> str:
        print(f"\nUser Feedback Required:\n{self.prompt}\n\nContext:\n{context}\n")
        while True:
            feedback = input("Please provide your feedback (or type 'done' if satisfied): ")
            if feedback.lower() == "done":
                break
            print(f"\nUser Feedback: {feedback}\n")
        return feedback

class WikipediaSearchTool:
    def __init__(self, chunk_size: int = 1000, num_chunks: int = 10):
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.chunker = TextChunker()

    def search_wikipedia(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        url = f"https://en.wikipedia.org/w/index.php?search={query}&title=Special:Search&fulltext=1"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        search_results = []
        for result in soup.find_all('li', class_='mw-search-result'):
            title = result.find('a').get_text()
            url = 'https://en.wikipedia.org' + result.find('a')['href']
            page_response = requests.get(url)
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            content = page_soup.find('div', class_='mw-parser-output').get_text()
            chunks = self.chunker.chunk_text(text=content, chunk_size=self.chunk_size, num_chunks=self.num_chunks)
            search_results.append({'title': title, 'url': url, 'chunks': chunks})
            if len(search_results) >= top_k:
                break

        return search_results

class SemanticFileSearchTool:
    def __init__(self, resources: List['Resources'], embed_model: str, embed_dim: int = 768, chunk_size: int = 1000, top_k: int = 3):
        self.embedder = Embed4All(embed_model)
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.chunker = TextChunker(text=None, chunk_size=chunk_size)
        self.file_embeddings = self.load_or_generate_file_embeddings(resources)

    def load_or_generate_file_embeddings(self, resources: List['Resources']) -> Dict[str, List[Dict[str, Any]]]:
        file_hash = self.get_file_hash(resources)
        pickle_file = f"file_embeddings_{file_hash}.pickle"
        if os.path.exists(pickle_file):
            self.load_embeddings(pickle_file)
        else:
            self.file_embeddings = self.generate_file_embeddings(resources)
            self.save_embeddings(pickle_file)
        return self.file_embeddings

    def get_file_hash(self, resources: List['Resources']) -> str:
        file_contents = "".join(sorted([resource.resource_path for resource in resources]))
        return hashlib.sha256(file_contents.encode()).hexdigest()

    def generate_file_embeddings(self, resources: List['Resources']) -> Dict[str, List[Dict[str, Any]]]:
        file_embeddings = {}
        for resource in resources:
            resource.chunk_resource(self.chunk_size)
            chunk_embeddings = [self.embedder.embed(chunk['text'], prefix='search_document') for chunk in resource.chunks]
            file_embeddings[resource.resource_path] = [
                {
                    'text': resource.contextualize_chunk(chunk),
                    'start': chunk['start'],
                    'end': chunk['end'],
                    'file': resource.resource_path,
                    'embedding': embedding
                }
                for chunk, embedding in zip(resource.chunks, chunk_embeddings)
            ]
        return file_embeddings

    def search(self, query: str) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed(query, prefix='search_query')
        scores = []
        for file_path, chunk_data in self.file_embeddings.items():
            for chunk in chunk_data:
                chunk_score = self.cosine_similarity(query_embedding, chunk['embedding'])
                scores.append((chunk, chunk_score))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_scores = sorted_scores[:self.top_k]
        result = []
        for chunk, score in top_scores:
            result.append({
                'file': chunk['file'],
                'text': chunk['text'],
                'score': score
            })
        return result

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def save_embeddings(self, pickle_file: str):
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.file_embeddings, f)

    def load_embeddings(self, pickle_file: str):
        with open(pickle_file, 'rb') as f:
            self.file_embeddings = pickle.load(f)


    @staticmethod
    def cosine_similarity(a, b) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



class MathFormulaParserTool:
    def __init__(self, resource: Resources, text: str = None):
        self.resource = resource
        self.text = text
        self.nlp = spacy.load("en_core_web_trf")

        # Define custom entity labels for mathematical symbols and numbers
        self.nlp.entity.add_label("MATH_SYMBOL")
        self.nlp.entity.add_label("NUMBER")

    def parse_formula(self, text: Optional[str] = None) -> str:
        if text is not None:
            self.text = text
        doc = self.nlp(self.text)

        # Define patterns for mathematical symbols and numbers
        math_symbols = ["=", "+", "-", "*", "/", "^", "(", ")"]
        number_pattern = r"\d+(\.\d+)?"

        # Find mathematical symbols and numbers in the text
        for token in doc:
            if token.text in math_symbols:
                token.ent_type_ = "MATH_SYMBOL"
            elif re.match(number_pattern, token.text):
                token.ent_type_ = "NUMBER"

        # Merge adjacent mathematical symbols and numbers into single entities
        entities = []
        prev_ent = None
        for ent in doc.ents:
            if prev_ent is not None and prev_ent.end == ent.start and prev_ent.label_ == ent.label_:
                prev_ent = Span(doc, prev_ent.start, ent.end, label=ent.label_)
            else:
                if prev_ent is not None:
                    entities.append(prev_ent)
                prev_ent = ent
        if prev_ent is not None:
            entities.append(prev_ent)

        # Wrap mathematical entities in LaTeX markdown tags
        formula_parts = []
        prev_end = 0
        for ent in entities:
            formula_parts.append(self.text[prev_end:ent.start])
            formula_parts.append(f"$${ent.text}$$")
            prev_end = ent.end
        formula_parts.append(self.text[prev_end:])

        # Join the formula parts into a single string
        parsed_formula = "".join(formula_parts)

        return parsed_formula



class SQLTool:
    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = None

    def connect(self):
        self.connection = sqlite3.connect(self.db_path)

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def create_category_table(self, category):
        if not self.connection:
            self.connect()

        create_table_query = f"CREATE TABLE IF NOT EXISTS {category} (id INTEGER PRIMARY KEY, text TEXT)"

        cursor = self.connection.cursor()
        cursor.execute(create_table_query)
        self.connection.commit()
        cursor.close()

    def insert_item(self, category, text):
        if not self.connection:
            self.connect()

        insert_query = f"INSERT INTO {category} (text) VALUES (?)"

        cursor = self.connection.cursor()
        cursor.execute(insert_query, (text,))
        self.connection.commit()
        cursor.close()

    def get_items(self, category):
        if not self.connection:
            self.connect()

        select_query = f"SELECT id, text FROM {category}"

        cursor = self.connection.cursor()
        cursor.execute(select_query)
        rows = cursor.fetchall()
        cursor.close()

        return rows

class Agent:
    def __init__(
        self,
        role: str,
        goal: str,
        tools: Optional[List[Any]] = None,
        verbose: bool = False,
        model: str = "mistral:instruct",
        max_iter: int = 25,
        max_rpm: Optional[int] = None,
        max_execution_time: Optional[int] = None,
        cache: bool = True,
        step_callback: Optional[Callable] = None,
        persona: Optional[str] = None,
        allow_delegation: bool = False,
        input_tasks: Optional[List["Task"]] = None,
        output_tasks: Optional[List["Task"]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.role = role
        self.goal = goal
        self.tools = tools or {}
        self.verbose = verbose
        self.model = model
        self.max_iter = max_iter
        self.max_rpm = max_rpm
        self.max_execution_time = max_execution_time
        self.cache = cache
        self.step_callback = step_callback
        self.persona = persona
        self.allow_delegation = allow_delegation
        self.input_tasks = input_tasks or []
        self.output_tasks = output_tasks or []
        self.interactions = []
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',
        )

    def execute_task(self, task: "Task", context: Optional[str] = None) -> str:
        messages = []
        if self.persona and self.verbose:
            messages.append({"role": "system", "content": f"{self.persona}"})
        system_prompt = f"You are a {self.role} with the goal: {self.goal}.\n"
        system_prompt += f"The expected output is: {task.expected_output}\n"
        messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": f"Your task is to {task.instructions}."})
        if context:
            messages.append({"role": "assistant", "content": f"Context from {task.context_agent_role}:\n{context}"})
        thoughts = []
        if task.tool_name in self.tools:
            tool = self.tools[task.tool_name]
            if isinstance(tool, (TextReaderTool, WebScraperTool)):
                text_chunks = tool.read_text() if isinstance(tool, TextReaderTool) else tool.scrape_text()
                for chunk in text_chunks:
                    thoughts.append(tool.resource.contextualize_chunk(chunk))
            elif isinstance(tool, SemanticFileSearchTool):
                query = "\n".join([c.output for c in task.context if c.output])
                relevant_chunks = tool.search(query)
                for chunk in relevant_chunks:
                    chunk_text = f"File: {chunk['file']}\nText: {chunk['text']}\nRelevance: {chunk['score']:.3f}"
                    thoughts.append(chunk_text)
            elif isinstance(tool, SemanticAnalysisTool):
                sentiment_result = tool.analyze_sentiment(context)
                thoughts.append(f"Sentiment Analysis Result: {sentiment_result}")
            elif isinstance(tool, NERExtractionTool):
                entities = tool.extract_entities(context)
                thoughts.append(f"Extracted Entities: {entities}")
            elif isinstance(tool, KnowledgeGraphTool):
                entities = tool.extract_entities()
                relationships = tool.extract_relationships(entities)
                graph = tool.construct_graph(entities, relationships)
                thoughts.append(f"Knowledge Graph:\n{graph}")
            if "sql_tool" in self.tools:
                sql_tool = self.tools["sql_tool"]
                if task.instructions.startswith("Create a category table"):
                    category = context.split(":")[-1].strip()
                    sql_tool.create_category_table(category)
                    return f"Category table '{category}' created successfully."
                elif task.instructions.startswith("Insert an item"):
                    category, item_text = context.split(":")[-1].strip().split(",")
                    sql_tool.insert_item(category.strip(), item_text.strip())
                    return f"Item inserted into category '{category}' successfully."
                elif task.instructions.startswith("Retrieve items from category"):
                    category = context.split(":")[-1].strip()
                    items = sql_tool.get_items(category)
                    if items:
                        result = f"Items in category '{category}':\n"
                        for item in items:
                            result += f"- ID: {item[0]}, Text: {item[1]}\n"
                        return result
                    else:
                        return f"No items found in category '{category}'."
            elif isinstance(tool, UserFeedbackTool):
                feedback = tool.request_feedback(context)
                thoughts.append(f"User Feedback: {feedback}")
            elif isinstance(tool, WikipediaSearchTool):
                search_results = tool.search_wikipedia(context)
                for result in search_results:
                    for chunk in result['chunks']:
                        chunk_text = f"Title: {result['title']}\nURL: {result['url']}\n{chunk['text']}"
                        thoughts.append(chunk_text)
        if thoughts:
            thoughts_prompt = "\n".join([thought for thought in thoughts])
            messages.append({"role": "user", "content": f"{thoughts_prompt}"})
        else:
            messages.append({"role": "user", "content": "No additional relevant information found."})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        result = response.choices[0].message.content
        self.log_interaction(messages, result)
        if self.step_callback:
            self.step_callback(task, result)
        return result

    def log_interaction(self, prompt, response):
        self.interactions.append({
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })

class Task:
    def __init__(
        self,
        instructions: str,
        expected_output: str,
        agent: Optional[Agent] = None,
        async_execution: bool = False,
        context: Optional[List["Task"]] = None,
        output_file: Optional[str] = None,
        callback: Optional[Callable] = None,
        human_input: bool = False,
        tool_name: Optional[str] = None,
        input_tasks: Optional[List["Task"]] = None,
        output_tasks: Optional[List["Task"]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.instructions = instructions
        self.expected_output = expected_output
        self.agent = agent
        self.async_execution = async_execution
        self.context = context or []
        self.output_file = output_file
        self.callback = callback
        self.human_input = human_input
        self.output = None
        self.context_agent_role = None
        self.tool_name = tool_name
        self.input_tasks = input_tasks or []
        self.output_tasks = output_tasks or []
        self.prompt_data = []

    def execute(self, context: Optional[str] = None) -> str:
        if not self.agent:
            raise Exception("No agent assigned to the task.")

        context_tasks = [task for task in self.context if task.output]
        if context_tasks:
            self.context_agent_role = context_tasks[0].agent.role
            original_context = "\n".join([f"{task.agent.role}: {task.output}" for task in context_tasks])

            if self.tool_name == 'semantic_search':
                query = "\n".join([task.output for task in context_tasks])
                context = query
            else:
                context = original_context

        prompt_details = self.prepare_prompt(context)
        self.prompt_data.append(prompt_details)

        result = self.agent.execute_task(self, context)
        self.output = result

        if self.output_file:
            with open(self.output_file, "w") as file:
                file.write(result)

        if self.callback:
            self.callback(self)

        return result


    def prepare_prompt(self, context):
        prompt = {
            "timestamp": datetime.now().isoformat(),
            "task_id": self.id,
            "instructions": self.instructions,
            "context": context,
            "expected_output": self.expected_output
        }
        return prompt

class researchANDGRAPH:
    def __init__(self, agents: List['Agent'], tasks: List['Task'], verbose: bool = False, log_file: str = "squad_log.json"):
        self.id = str(uuid.uuid4())
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose
        self.log_file = log_file
        self.log_data = []
        self.llama_logs = []

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> str:
        context = ""
        for task in self.tasks:
            if self.verbose:
                print(f"Starting Task:\n{task.instructions}")

            self.log_data.append({
                "timestamp": datetime.now().isoformat(),
                "type": "input",
                "agent_role": task.agent.role,
                "task_name": task.instructions,
                "task_id": task.id,
                "content": task.instructions
            })

            output = task.execute(context=context)
            task.output = output

            if self.verbose:
                print(f"Task output:\n{output}\n")

            self.log_data.append({
                "timestamp": datetime.now().isoformat(),
                "type": "output",
                "agent_role": task.agent.role,
                "task_name": task.instructions,
                "task_id": task.id,
                "content": output
            })

            self.llama_logs.extend(task.agent.interactions)

            context += f"Task:\n{task.instructions}\nOutput:\n{output}\n\n"

            self.handle_tool_logic(task, context)

        self.save_logs()
        self.save_llama_logs()

        return context


    def handle_tool_logic(self, task, context):
        if task.tool_name in task.agent.tools:
            tool = task.agent.tools[task.tool_name]
            if isinstance(tool, (TextReaderTool, WebScraperTool, SemanticFileSearchTool)):
                text_chunks = self.handle_specific_tool(task, tool)
                for i, chunk in enumerate(text_chunks, start=1):
                    self.log_data.append({
                        "timestamp": datetime.now().isoformat(),
                        "type": "text_chunk",
                        "task_id": task.id,
                        "chunk_id": i,
                        "text": chunk['text'],
                        "start": chunk.get('start', 0),
                        "end": chunk.get('end', len(chunk['text'])),
                        "file": chunk.get('file', '')
                    })

            if isinstance(tool, SemanticAnalysisTool):
                sentiment_result = tool.analyze_sentiment(task.output)
                self.log_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "sentiment_analysis",
                    "task_id": task.id,
                    "content": sentiment_result
                })
                context += f"Sentiment Analysis Result: {sentiment_result}\n\n"

            if isinstance(tool, NERExtractionTool):
                entities = tool.extract_entities(task.output)
                self.log_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "ner_extraction",
                    "task_id": task.id,
                    "content": [ent['text'] for ent in entities]
                })
                context += f"Extracted Entities: {[ent['text'] for ent in entities]}\n\n"

    def handle_specific_tool(self, task, tool):
        if isinstance(tool, SemanticFileSearchTool):
            query = "\n".join([c.output for c in task.context if c.output])
            return tool.search(query)
        else:
            return tool.read_text() if isinstance(tool, TextReaderTool) else tool.scrape_text()

    def save_llama_logs(self):
        with open(("qa_interactions" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"), "w") as file:
            json.dump(self.llama_logs, file, indent=2)

    def save_logs(self):
        with open(self.log_file, "w") as file:
            json.dump(self.log_data, file, indent=2)


def mainflow():


# TOOL_RESOURCES

    text_resource = Resources('text', "inputs/cyberanimism_clean.txt", "Here are your thoughts on the statement '{chunk}' from the file '{file}' (start: {start}, end: {end}): ")
    pdf_resource = Resources('pdf', "inputs/book1.pdf", "The following excerpt is from the PDF '{file}' (start: {start}, end: {end}):\n{chunk}")
    web_resource = Resources('web', "https://www.dvdtalk.com/reviews/23083/what-the-bleepdown-the-rabbit-hole-quantum-edition/", "The following content is scraped from the web page '{file}':\n{chunk}")
    secret_poetry_resource = Resources('text', "inputs/bloodypoetry.txt", "Here is a secret poem for you:\n{chunk}")


# TOOLS

    text_reader_tool = TextReaderTool(text_resource, chunk_size=128, num_chunks=8)
    web_scraper_tool = WebScraperTool(web_resource, chunk_size=512, num_chunks=8)
    semantic_search_tool = SemanticFileSearchTool(resources=[pdf_resource, text_resource, secret_poetry_resource], embed_model='nomic-embed-text-v1.5.f16.gguf', chunk_size=256, top_k=4)
 

# AGENTS

    researcher = Agent(
        role='Researcher',
        goal='Analyze the provided text and extract relevant information.',
        persona="""You are a renowned Content Strategist, known for your insightful and engaging articles. You transform complex concepts into compelling narratives.""",
        tools={"text_reader": text_reader_tool},
        model="phi3",
        verbose=True
    )

    web_analyzer = Agent(
        role='Web Analyzer',
        goal='Analyze the scraped web content and provide a summary.',
        tools={"web_scraper": web_scraper_tool},
        model="adrienbrault/nous-hermes2pro-llama3-8b:q4_K_M",
        verbose=True
    )


    semantic_searcher = Agent(
        role='Semantic Searcher',
        goal='Perform semantic searches on a corpus of files to find relevant information.',
        persona='You are an expert in semantic search and information retrieval.',
        tools={'semantic_search': semantic_search_tool},
        verbose=True
    )

    summarizer = Agent(
        role='Summarizer',
        persona="""You are a skilled Data Analyst with a knack for distilling complex streams of thought into factual information as dense summaries. """,
        goal='Compile a summary report based on the extracted information. Facts start as thoughts, and thoughts are the seeds your next action. Provide 1500~ words of summary.',
        verbose=True
    )


# TASKS

    txt_task = Task(
        instructions="Analyze the provided text and identify key insights and patterns.",
        expected_output="A list of key insights and patterns found in the text.",
        agent=researcher,
        output_file='txt_analyzed.txt',
        tool_name="text_reader",
    )

    web_task = Task(
        instructions="Scrape the content from the provided URL and provide a summary.",
        expected_output="A summary of the scraped web content.",
        agent=web_analyzer,
        tool_name="web_scraper",
        output_file='web_task_output.txt',
    )

    search_task = Task(
        instructions='Search the provided files for information relevant to the given query.',
        expected_output='A list of relevant files with their similarity scores.',
        agent=semantic_searcher,
        tool_name='semantic_search',
        context=[txt_task],
    )

    summary = Task(
        instructions="Using the insights from the researcher and web analyzer, compile a summary report.",
        expected_output="A well-structured summary report based on the extracted information.",
        agent=summarizer,
        context=[search_task, txt_task, web_task],
        output_file='summarytask_output.txt',
    )

# FLOW SEQUENCE

    flow = researchANDGRAPH(
        agents=[researcher, semantic_searcher, web_analyzer, summarizer, semantic_searcher],
        tasks=[txt_task, search_task, txt_task, web_task, summary],
        verbose=True,
        log_file="squad_goals" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    )

    result = flow.run()
    print(f"Final output:\n{result}")

if __name__ == "__main__":
    mainflow()
