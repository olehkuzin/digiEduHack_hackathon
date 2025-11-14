from openai import OpenAI
import os

# Load API key for the Featherless endpoint from environment variables
API_KEY = os.getenv("META_API_KEY")

# Initialize OpenAI-compatible client for the Featherless API
client = OpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=API_KEY,
)

# System prompt template for the analytical education-policy assistant
prompt = """
<prompt>
  <role>
    You are an analytical assistant specializing in education policy and school systems.
  </role>
  <task>
    Read the provided text and extract all information related to schools, schooling, and the education system, including any reforms, trends, policies, funding changes, quality indicators, and learning outcomes.
  </task>
  <analysis>
    Focus only on content directly connected to education and schools, and determine for each described change whether it represents an improvement or a deterioration, and for which groups (students, teachers, parents, schools, or regions).
  </analysis>
  <output_requirements>
    Write a concise summary of the situation and its evolution in exactly 5–6 sentences, clearly describing the main changes, their direction (better or worse), and their key causes or consequences.
    Do not use bullet points, quotes, or formatting; write in clear, neutral, and accessible language that a non-expert can understand.
  </output_requirements>
</prompt>
"""


def call_agent(prompt: str, text: str) -> str:
    """
    Send text to the Llama-3.3-70B model using the Featherless API and return its response.

    Parameters
    ----------
    prompt : str
        System prompt defining the behavior and constraints of the model.
    text : str
        User-provided text to be analyzed.

    Returns
    -------
    str
        The model-generated summary or analytical output.
    """
    # Send the system and user messages to the model deterministically
    response = client.chat.completions.create(
        model='meta-llama/Llama-3.3-70B-Instruct',
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        temperature=0
    )

    # Extract and return the model's output text
    return response.choices[0].message.content
