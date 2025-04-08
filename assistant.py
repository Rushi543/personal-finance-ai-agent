from uagents import Agent, Context, Model, Protocol
from uagents.setup import fund_agent_if_low
import google.generativeai as genai
from Finance import FinanceAgent
import json
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
os.makedirs("outputs", exist_ok=True)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')
ai_assistant = Agent(
    name='AI Assistant',
    port=5052,
    mailbox=True
)

fund_agent_if_low(ai_assistant.wallet.address())

class Message(Model):
    message : str

user_query_protocol = Protocol()

finance_agent=FinanceAgent()
def categorize_query(user_input_text):
    """Use Gemini to categorize a transaction based on description and amount"""
    prompt = f"""You are a financial assistant AI. Your task is to read the given text and classify it into one of the following categories based on its primary intent or content:

    1.Manage Transactions – If the text refers to recording, or organizing financial transactions such as purchases, payments, or transfers.

    2.Budget Setting – If the text involves setting up or adjusting a budget, planning future expenses, or distributing income across categories (not budget recommendations).

    3.Financial Analysis – If the text involves analyzing, calulating financial data, interpreting trends, assessing financial performance, or evaluating spending habits.

    4.Financial Advice – If the text seeks or provides guidance on investments, saving strategies, debt management, or any financial decision-making support.

    5.Other – If the text does not clearly fall under any of the above categories.

    Put all the questions or advices which doesn't involve manipulating database in Financial Advice or Other
    Input:
    "{user_input_text}"

    Output:
    Return only the category name from the list above that best describes the input text.
    """
    
    response = model.generate_content(prompt)
    category = response.text.strip()
    return category

def extract_transaction(query):
    prompt = f"""
    
    You are a smart financial assistant. Extract transaction details using the 50/30/20 rule.

    Input: "{query}"

    Return JSON with:
    - date: YYYY-MM-DD format based on the transaction date
      * Convert relative terms (today, yesterday) to actual dates
      * Use today's date ({datetime.now().strftime('%Y-%m-%d')}) as reference
      * Return null if no date is provided
    - amount: numeric value (negative for money spent, positive for money received)
    - description: brief description of the transaction
    - budget_type: categorized as:
        * "needs" (essential expenses: housing, utilities, groceries, healthcare, etc.)
        * "wants" (non-essential: dining out, entertainment, shopping, etc.)
        * "savings_or_debt" (investments, debt payments, emergency fund)
        * "income" (salary, gifts received, refunds, etc.)
        * "unknown" (when category cannot be determined)

    Example output format:
    {{
    "date": "2025-04-08",
    "amount": -45.67,
    "description": "Grocery shopping at Whole Foods",
    "budget_type": "needs"
    }}

    Only return the JSON output in the specified format.
    """

    response = model.generate_content(prompt)
    details = response.text.strip()
    cleaned = details.strip("`").replace("json", "").strip()
    data = json.loads(cleaned)
    return data

def extract_budget(query):
    prompt = f"""
    
    You are a smart financial assistant. Extract budget details.
    Categorize into one of these categories: Food, Transportation, Housing, Entertainment, Shopping, Utilities, Healthcare, Education, Travel, Income, Other
    
    Input: "{query}"

    Return JSON with:
    - amount: numeric value
    - category: category of the budget plannning

    Example output format:
    {{
    "amount": 50,
    "category": "Food"
    }}

    Only return the JSON output in the specified format.
    """

    response = model.generate_content(prompt)
    details = response.text.strip()
    cleaned = details.strip("`").replace("json", "").strip()
    data = json.loads(cleaned)
    return data

def handlequery(query):
    category=categorize_query(query)
    if category=="Manage Transactions":
        transaction=extract_transaction(query)
        finance_agent.add_transaction(date=transaction['date'],amount=transaction['amount'],description=transaction['description'],type=transaction['budget_type'])
        return "Transactions updated!"
    elif category=="Budget Setting":
        budget=extract_budget(query)
        finance_agent.set_budget_goal(category=budget['category'],amount=budget['amount'])
        return "Budget Plan updated!"
    elif category=='Financial Analysis':
        result, fig, analysis_code=finance_agent.analyze_data(query=query)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"outputs/fig_{timestamp}.png"
        if fig:
            fig.write_image(file_path)
        return result
    elif category=='Financial Advice':
        return finance_agent.get_financial_advice(query=query)
    else:
        response = model.generate_content(query)
        return response.text.strip()

@ai_assistant.on_event('startup')
async def startup_handler(ctx: Context):
    ctx.logger.info(f'My name is {ctx.agent.name} and my address is {ctx.agent.address}')

@user_query_protocol.on_message(model=Message, replies=None)
async def handle_user_query(ctx: Context, sender: str, msg: Message):
    ctx.logger.info(f"Received query from {sender}: {msg.message}")
    response=handlequery(msg.message)
    await ctx.send(sender, Message(message=response)) 

ai_assistant.include(user_query_protocol)

if __name__ == "__main__":
    ai_assistant.run()