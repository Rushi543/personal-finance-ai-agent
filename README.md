# personal-finance-ai-agent
![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:domain/finance](https://img.shields.io/badge/domain-finance-4CAF50)

# 💸 AI Finance Assistant
agent address: agent1qg7hszcwltxxms8a6uvesqztnytt3pngfzsvtzu8acfpc6gs0mglvwruhs0

**Description**:  
This AI Agent helps users analyze personal financial transactions, identify spending habits, and generate insights using natural language queries. It supports category-wise breakdowns, trend visualizations, and budgeting guidance.

The assistant can parse free-form queries like:
- “Where am I spending too much money?”
- “Show me my top 3 expense categories.”
- “What was my income trend last month?”
-  "I spent 60$ on groceries yesterday"

It automatically analyzes structured financial data from Excel, CSV, or user input and replies with graph and insights.

---

## 🚀 Capabilities

- Understands **natural language** financial queries  
- Supports **category classification** (e.g., wants vs needs)  
- Parses and extracts financial info from **free-text inputs**  
- Plots **interactive graphs** using Plotly  
- Generates and stores **user-specific Excel reports**  
- Works with **chat history and recent memory context**  
- Saves analysis results

---

## 🧠 Use Cases

  - Understand monthly spending patterns using natural queries
  - Automatically categorize expenses from uploaded Excel files
  - Track income vs. expenses with time-based trends
  - Generate personal finance reports per user in .xlsx format
  - Suggest budget improvements using Wants vs Needs logic

---

## 📥 Input Data Model

```python
class Message(Model):
    message : str
```

## 🚀 Usage Example

Copy and paste the following code into a new **Blank agent** to interact with this AI Finance Assistant agent.

```python
from uagents import Agent, Context, Model

agent = Agent()

AI_AGENT_ADDRESS = "agent1qg7hszcwltxxms8a6uvesqztnytt3pngfzsvtzu8acfpc6gs0mglvwruhs0"

class Message(Model):
    message : str

@agent.on_event("startup")
async def handle_startup(ctx: Context):
    """Send the query to the AI agent on startup."""
    await ctx.send(AI_AGENT_ADDRESS, Message(
        message="I bought groceries of 50 yesterday"
    ))
    ctx.logger.info(f"Query sent to Finance Assistant agent")

@agent.on_message(model=Message)
async def handle_response(ctx: Context, sender: str, msg: Message):
    """Receive and display the response."""
    ctx.logger.info(f"Received response from {sender}: {msg.message}")

if __name__ == "__main__":
    agent.run()
```
## Local Agent
- Install the necessary packages:

```pip install requests uagents```

To interact with this agent from a local agent instead, replace agent = Agent() in the above with:

```
agent = Agent(
    name="user",
    endpoint="http://localhost:8000/submit",
)
```

Run the agent:

```python agent.py```

🚀 Local Setup

- Install dependencies

Ensure Python 3.8+ is installed. Then:
```
pip install -r requirements.txt
```
- Create a .env file

Add your Gemini API Key to the root .env file:
```
GEMINI_API_KEY=your_actual_api_key_here
```
- Run the app
```
python assistant.py
python agent.py
```
