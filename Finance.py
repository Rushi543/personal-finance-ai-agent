import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, date as dt_date, timedelta
import google.generativeai as genai
import streamlit as st
import plotly.express as px
import json
import uuid
import time

GOOGLE_API_KEY = "AIzaSyD1a7z1Fr3cyAyFdYp-egWuipfcCD71KTQ"  
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')
class FinanceAgent:
    def __init__(self, file_path='transactions.xlsx', memory_path='agent_memory.json'):
        # Set up data storage
        self.file_path = file_path
        self.memory_path = memory_path
        
        # Load transaction data
        if os.path.exists(file_path):
            self.df = pd.read_excel(file_path)
        else:
            self.df = pd.DataFrame(columns=['date', 'amount', 'description', 'category', 'transaction_id'])
            self.df.to_excel(file_path, index=False)

        # Ensure date is datetime and each transaction has an ID
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        if 'transaction_id' not in self.df.columns:
            self.df['transaction_id'] = [str(uuid.uuid4()) for _ in range(len(self.df))]
            self.df.to_excel(file_path, index=False)
        
        # Load agent memory
        self.memory = self._load_memory()
        
        # Chat history for conversational context
        if 'chat_history' not in self.memory:
            self.memory['chat_history'] = []
            
        # User preferences and insights
        if 'user_preferences' not in self.memory:
            self.memory['user_preferences'] = {
                'budget_goals': {},
                'saving_targets': {},
                'categories_to_watch': []
            }
            
        # Agent insights and observations
        if 'agent_insights' not in self.memory:
            self.memory['agent_insights'] = []
            
        # Save initialized memory
        self._save_memory()
        
        print("Finance Agent initialized with data and memory")

    def _load_memory(self):
        """Load agent memory from JSON file"""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_memory(self):
        """Save agent memory to JSON file"""
        with open(self.memory_path, 'w') as f:
            json.dump(self.memory, f)

    def categorize_transaction(self, description, amount):
        """Categorize a transaction using AI"""
        # Include past categories to help with consistency
        recent_similar = self._find_similar_transactions(description)
        similar_examples = "\n".join([f"Description: {row['description']}, Amount: ${row['amount']}, Category: {row['category']}" 
                                 for _, row in recent_similar.iterrows()])
        
        prompt = f"""
        Categorize this transaction into one of these categories: Food, Transportation, Housing, Entertainment, Shopping, Utilities, Healthcare, Education, Travel, Income, Other
        
        Transaction: {description}
        Amount: ${amount}
        
        Examples of similar past transactions:
        {similar_examples if not recent_similar.empty else "No similar transactions found."}
        
        Return only the category name without any explanation.
        """
        response = model.generate_content(prompt)
        category = response.text.strip()
        return category

    def _find_similar_transactions(self, description, limit=3):
        """Find similar transactions in history"""
        if self.df.empty:
            return pd.DataFrame()
            
        # This is a simple implementation. For production, consider using embeddings or better similarity metrics
        description_lower = description.lower()
        
        # Find transactions containing similar words
        words = set(description_lower.split())
        mask = self.df['description'].str.lower().apply(
            lambda x: any(word in str(x).lower() for word in words if len(word) > 3)
        )
        
        return self.df[mask].tail(limit)

    def add_transaction(self, date, amount, description,type):
        """Add a new transaction with AI categorization"""
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        elif isinstance(date, dt_date):
            date = datetime.combine(date, datetime.min.time())

        # Generate a unique ID for this transaction
        transaction_id = str(uuid.uuid4())
        
        # Categorize using AI
        category = self.categorize_transaction(description, amount)
        
        new_row = pd.DataFrame({
            'date': [date],
            'amount': [float(amount)],
            'description': [description],
            'category': [category],
            'budget_type':[type],
            'transaction_id': [transaction_id]
        })

        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df.to_excel(self.file_path, index=False)

        # Update agent memory with this transaction
        self._update_memory_with_transaction(date, amount, description, category, transaction_id)
        
        # Generate insights asynchronously (in a real app, this would be a background task)
        self._generate_new_insights()

        return category, transaction_id

    def _update_memory_with_transaction(self, date, amount, description, category, transaction_id):
        """Update agent memory with new transaction details"""
        if 'recent_transactions' not in self.memory:
            self.memory['recent_transactions'] = []
            
        # Add to recent transactions (limited to last 20)
        self.memory['recent_transactions'].append({
            'date': date.strftime('%Y-%m-%d'),
            'amount': float(amount),
            'description': description,
            'category': category,
            'transaction_id': transaction_id
        })
        
        # Keep only last 20 transactions in memory
        if len(self.memory['recent_transactions']) > 20:
            self.memory['recent_transactions'] = self.memory['recent_transactions'][-20:]
            
        # Update category spending trends
        if 'category_trends' not in self.memory:
            self.memory['category_trends'] = {}
            
        if category not in self.memory['category_trends']:
            self.memory['category_trends'][category] = {
                'count': 0,
                'total': 0,
                'average': 0
            }
            
        self.memory['category_trends'][category]['count'] += 1
        self.memory['category_trends'][category]['total'] += float(amount)
        self.memory['category_trends'][category]['average'] = (
            self.memory['category_trends'][category]['total'] / 
            self.memory['category_trends'][category]['count']
        )
        
        # Save updated memory
        self._save_memory()

    def _generate_new_insights(self):
        """Generate new insights based on latest transactions"""
        # Get recent spending habits
        recent_df = self.df[self.df['date'] >= datetime.now() - timedelta(days=30)].copy()
        
        if recent_df.empty:
            return
            
        # Format data for the LLM
        recent_spending = recent_df[recent_df['amount'] < 0].groupby('category')['amount'].sum().reset_index()
        recent_spending['amount'] = recent_spending['amount'].abs()
        
        top_categories = recent_spending.sort_values('amount', ascending=False).head(3)
        top_categories_text = ", ".join([f"{row['category']}: ${row['amount']:.2f}" for _, row in top_categories.iterrows()])
        
        # Generate insights with LLM
        prompt = f"""
        As a financial AI agent, generate 1-2 new insights based on these recent spending patterns:
        
        Top spending categories in the last 30 days:
        {top_categories_text}
        
        Total recent expenses: ${recent_df[recent_df['amount'] < 0]['amount'].sum() * -1:.2f}
        
        Existing insights I've already shared:
        {self.memory['agent_insights'][-3:] if len(self.memory['agent_insights']) > 0 else "None yet"}
        
        Generate a single, specific, actionable insight that's different from previous ones. 
        Keep it under 100 words and focus on practical advice.
        """
        
        try:
            response = model.generate_content(prompt)
            new_insight = response.text.strip()
            
            # Only add if we have a valid insight
            if len(new_insight) > 10:
                self.memory['agent_insights'].append(new_insight)
                # Keep only latest 10 insights
                if len(self.memory['agent_insights']) > 10:
                    self.memory['agent_insights'] = self.memory['agent_insights'][-10:]
                self._save_memory()
        except Exception as e:
            print(f"Error generating insights: {e}")

    def analyze_data(self, query):
        """Analyze financial data based on natural language query"""
        # Add recent queries to memory for context
        if 'recent_queries' not in self.memory:
            self.memory['recent_queries'] = []
            
        self.memory['recent_queries'].append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'query': query
        })
        
        # Keep memory manageable
        if len(self.memory['recent_queries']) > 10:
            self.memory['recent_queries'] = self.memory['recent_queries'][-10:]
            
        self._save_memory()
        
        # Add the user's query to chat history
        self.add_to_chat_history('user', query)
        
        # Generate code for analysis with contextual understanding
        prompt = f"""
        You are a code-generating assistant for a financial analysis agent.

        Write Python code to analyze the following financial transaction data based on this query:
        "{query}"

        The data is in a pandas DataFrame called `df` with these columns:
        - `date`: datetime64[ns]
        - `amount`: float (positive for income, negative for expenses)
        - `description`: string
        - `category`: string
        - `transaction_id`: string

        Recent queries from the user (for context):
        {json.dumps(self.memory.get('recent_queries', [])[-3:], indent=2)}

        This is a Streamlit app. Follow these rules:
        1. Use only pandas, plotly.express (px), standard Python libs.
        2. After groupby, always call .reset_index().
        3. Do not use print(), fig.show().
        4. Store your plot in `fig`, and your insight string in `result`.
        5. Return only executable code (no markdown/code blocks).
        6. If creating time-based analysis, always sort by date.
        7. Convert date columns to proper datetime types if needed.
        8. Handle edge cases like empty DataFrames gracefully.
        """
        response = model.generate_content(prompt)
        analysis_code = response.text.strip()

        if analysis_code.startswith("```python"):
            analysis_code = analysis_code.split("```python")[1]
        if analysis_code.endswith("```"):
            analysis_code = analysis_code.split("```")[0]

        local_vars = {'df': self.df.copy(), 'px': px, 'pd': pd, 'plt': plt, 'datetime': datetime}
        try:
            exec(analysis_code, {}, local_vars)
            result = local_vars.get('result', "No insights were generated.")
            fig = local_vars.get('fig', None)
            
            # Add the result to chat history
            self.add_to_chat_history('agent', result)
            
            return result, fig, analysis_code
        except Exception as e:
            error_message = f"Error during analysis: {str(e)}"
            self.add_to_chat_history('agent', error_message)
            return error_message, None, analysis_code

    def add_to_chat_history(self, speaker, message):
        """Add message to chat history"""
        self.memory['chat_history'].append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'speaker': speaker,
            'message': message
        })
        
        # Keep chat history manageable (last 50 messages)
        if len(self.memory['chat_history']) > 50:
            self.memory['chat_history'] = self.memory['chat_history'][-50:]
            
        self._save_memory()

    def get_budget_recommendation(self):
        """Generate a personalized budget recommendation"""
        # Get monthly income
        monthly_income = self.df[self.df['amount'] > 0]['amount'].sum()
        
        # Get monthly expenses by category
        expenses = self.df[self.df['amount'] < 0].copy()
        expenses['amount'] = expenses['amount'].abs()  # Convert to positive for readability
        monthly_expenses = expenses.groupby('category')['amount'].sum().reset_index()
        
        # Format data for the LLM
        expense_summary = "\n".join([f"- {row['category']}: ${row['amount']:.2f}" for _, row in monthly_expenses.iterrows()])
        
        prompt = f"""
        As a financial advisor, create a personalized monthly budget based on this data:

        Monthly Income: ${monthly_income:.2f}
        
        Current Monthly Expenses:
        {expense_summary}
        
        Total Expenses: ${expenses['amount'].sum():.2f}
        
        Create a recommended budget allocation using the 50/30/20 rule (50% needs, 30% wants, 20% savings)
        or another appropriate framework. Specify dollar amounts for each category and provide 2-3 specific
        suggestions for improving financial health.
        
        Format the response in markdown.
        """
        
        response = model.generate_content(prompt)
        budget_plan = response.text.strip()
        
        # Add this to the agent's memory
        self.add_to_chat_history('agent', f"Generated budget recommendation: {budget_plan[:100]}...")
        
        return budget_plan

    def get_financial_advice(self, query=None):
        """Get personalized financial advice based on transaction history and query"""
        self.df['amount'] = pd.to_numeric(self.df['amount'], errors='coerce')
        
        # Basic financial summary
        income = self.df[self.df['amount'] > 0]['amount'].sum()
        expenses = self.df[self.df['amount'] < 0]['amount'].sum() * -1
        savings_rate = ((income - expenses) / income * 100) if income > 0 else 0
        
        categories = self.df.groupby('category')['amount'].sum().sort_values()
        top_expenses = categories[categories < 0].abs().nlargest(3)
        
        if top_expenses.empty:
            top_expense_text = "No major expenses found."
        else:
            top_expense_text = ', '.join([f"{cat}: ${abs(amt):.2f}" for cat, amt in top_expenses.items()])
        
        # Get recent insights from memory
        recent_insights = "\n".join(self.memory.get('agent_insights', [])[-3:])
        
        # Format chat history for context
        recent_chat = self.memory.get('chat_history', [])[-5:]
        chat_context = "\n".join([f"{msg['speaker']}: {msg['message'][:100]}..." for msg in recent_chat])
        
        summary = f"""
        Financial Summary:
        - Total income: ${income:.2f}
        - Total expenses: ${expenses:.2f}
        - Savings rate: {savings_rate:.1f}%
        - Top expense categories: {top_expense_text}
        
        Recent insights:
        {recent_insights}
        
        Recent conversation context:
        {chat_context}
        """

        prompt = f"""
        You are a helpful financial advisor agent with memory of past interactions.
        Based on this summary and the user's query, provide personalized financial advice.

        {summary}

        User query: {query if query else 'Give me general financial advice based on my situation'}

        Respond in a conversational tone with 3-5 actionable tips. 
        Reference specific transactions or patterns where relevant.
        Avoid generic advice - make it personalized based on the data.
        """
        
        response = model.generate_content(prompt)
        advice = response.text.strip()
        
        # Add to chat history
        if query:
            self.add_to_chat_history('user', query)
        self.add_to_chat_history('agent', advice)
        
        return advice

    def set_budget_goal(self, category, amount):
        """Set a budget goal for a category"""
        self.memory['user_preferences']['budget_goals'][category] = float(amount)
        self._save_memory()
        
        # Add to chat
        self.add_to_chat_history('agent', f"Budget goal set: {category} - ${amount:.2f}/month")
        
        return True
        
    def check_budget_progress(self):
        """Check progress against budget goals"""
        if not self.memory['user_preferences']['budget_goals']:
            return "No budget goals have been set yet."
            
        # Get current month's spending by category
        now = datetime.now()
        start_of_month = datetime(now.year, now.month, 1)
        
        current_spending = self.df[
            (self.df['date'] >= start_of_month) & 
            (self.df['amount'] < 0)
        ].copy()
        
        current_spending['amount'] = current_spending['amount'].abs()  # Make positive for readability
        spending_by_category = current_spending.groupby('category')['amount'].sum().to_dict()
        
        # Compare with goals
        progress = []
        for category, goal in self.memory['user_preferences']['budget_goals'].items():
            spent = spending_by_category.get(category, 0)
            percentage = (spent / goal) * 100 if goal > 0 else 0
            status = "Over budget" if percentage > 100 else "On track"
            
            progress.append({
                'category': category,
                'goal': goal,
                'spent': spent,
                'percentage': percentage,
                'status': status
            })
            
        # Format as dataframe for display
        progress_df = pd.DataFrame(progress)
        
        # Generate narrative
        prompt = f"""
        As a financial agent, give a brief assessment of the user's budget progress this month.
        
        Budget goals and progress:
        {progress_df.to_string() if not progress_df.empty else "No data available"}
        
        Keep your response brief (2-3 sentences) and mention the categories that need attention.
        """
        
        try:
            response = model.generate_content(prompt)
            narrative = response.text.strip()
        except:
            narrative = "Here's your current budget progress for the month."
            
        return progress_df, narrative

    def detect_unusual_transactions(self):
        """Detect potentially unusual transactions"""
        if len(self.df) < 5:  # Need some history to detect unusual patterns
            return None, "Not enough transaction history to detect unusual patterns."
            
        # For each category, find transactions that are significantly higher than average
        unusual = []
        
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category].copy()
            
            if len(cat_data) >= 3:  # Need at least 3 transactions to establish a pattern
                cat_data['amount'] = cat_data['amount'].abs()  # Work with absolute values
                avg = cat_data['amount'].mean()
                std = cat_data['amount'].std()
                
                # Transactions more than 2 standard deviations from mean are unusual
                threshold = avg + (2 * std)
                
                unusual_in_category = cat_data[cat_data['amount'] > threshold].copy()
                unusual.append(unusual_in_category)
        
        if not unusual:
            return None, "No unusual transactions detected."
            
        unusual_df = pd.concat(unusual, ignore_index=True) if unusual else pd.DataFrame()
        
        if unusual_df.empty:
            return None, "No unusual transactions detected."
            
        # Generate explanation
        prompt = f"""
        As a financial agent, explain these potentially unusual transactions:
        
        {unusual_df[['date', 'description', 'amount', 'category']].to_string()}
        
        Keep your explanation brief (2-3 sentences) and helpful.
        """
        
        try:
            response = model.generate_content(prompt)
            explanation = response.text.strip()
        except:
            explanation = "I've detected some transactions that appear unusual based on your spending patterns."
            
        return unusual_df, explanation

    def create_saving_plan(self, goal_amount, timeframe_months):
        """Create a personalized saving plan"""
        # Calculate average monthly savings capacity
        monthly_income = self.df[self.df['amount'] > 0]['amount'].sum() / max(1, len(self.df['date'].dt.month.unique()))
        monthly_expenses = self.df[self.df['amount'] < 0]['amount'].sum() * -1 / max(1, len(self.df['date'].dt.month.unique()))
        
        current_savings_capacity = monthly_income - monthly_expenses
        
        # Required monthly savings for goal
        required_monthly = float(goal_amount) / float(timeframe_months)
        
        # Categories with potential for savings
        expense_by_category = self.df[self.df['amount'] < 0].copy()
        expense_by_category['amount'] = expense_by_category['amount'].abs()
        category_spending = expense_by_category.groupby('category')['amount'].agg(['sum', 'count']).reset_index()
        
        prompt = f"""
        Create a realistic saving plan to save ${goal_amount:.2f} in {timeframe_months} months.
        
        Current financial situation:
        - Monthly income: ${monthly_income:.2f}
        - Monthly expenses: ${monthly_expenses:.2f}
        - Current savings capacity: ${current_savings_capacity:.2f}/month
        - Required savings for goal: ${required_monthly:.2f}/month
        
        Spending by category:
        {category_spending.to_string()}
        
        Create a step-by-step saving plan that includes:
        1. Whether the goal is realistic given the timeframe
        2. Specific categories where spending could be reduced
        3. Estimated monthly saving amount
        4. Any additional income strategies if needed
        5. A monthly breakdown of the saving plan
        
        Format the response in markdown.
        """
        
        response = model.generate_content(prompt)
        saving_plan = response.text.strip()
        
        # Store this goal in memory
        if 'saving_goals' not in self.memory:
            self.memory['saving_goals'] = []
            
        self.memory['saving_goals'].append({
            'date_created': datetime.now().strftime('%Y-%m-%d'),
            'goal_amount': float(goal_amount),
            'timeframe_months': int(timeframe_months),
            'monthly_target': required_monthly
        })
        
        self._save_memory()
        self.add_to_chat_history('agent', f"Created savings plan for ${goal_amount:.2f}")
        
        return saving_plan

    def chat(self, user_message):
        """Handle open-ended chat about finances"""
        # Add user message to history
        self.add_to_chat_history('user', user_message)
        
        # Format recent chat history for context
        chat_context = self.memory.get('chat_history', [])[-10:]
        formatted_chat = "\n".join([f"{msg['speaker']}: {msg['message']}" for msg in chat_context])
        
        # Get financial context
        income = self.df[self.df['amount'] > 0]['amount'].sum()
        expenses = self.df[self.df['amount'] < 0]['amount'].sum() * -1
        
        categories = self.df.groupby('category')['amount'].sum()
        top_expenses = categories[categories < 0].abs().nlargest(3)
        top_expense_text = ', '.join([f"{cat}: ${abs(amt):.2f}" for cat, amt in top_expenses.items()])
        
        financial_context = f"""
        Financial summary:
        - Total income: ${income:.2f}
        - Total expenses: ${expenses:.2f}
        - Top expenses: {top_expense_text}
        """
        
        # Check if this looks like a specific financial question
        query_prompt = f"""
        Analyze this user message and determine the user's intent:
        "{user_message}"
        
        Choose ONE of these categories that best matches their intent:
        1. General Chat - Just conversational, greeting, or non-specific question
        2. Analysis Question - Asking about specific spending patterns or financial analysis
        3. Budget Question - Question about budgeting or spending limits
        4. Advice Question - Asking for financial advice
        5. Action Request - User wants to set a goal, plan, or take action
        
        Return ONLY the category number (1-5) without explanation.
        """
        
        try:
            intent_response = model.generate_content(query_prompt)
            intent = intent_response.text.strip()
        except:
            intent = "1"  # Default to general chat
        
        # Generate response based on intent
        if intent == "2":  # Analysis question
            # Handle as analysis query
            result, _, _ = self.analyze_data(user_message)
            return result
        elif intent == "3" or intent == "4":  # Budget or advice question
            # Get detailed financial advice
            return self.get_financial_advice(user_message)
        elif intent == "5":  # Action request
            action_prompt = f"""
            The user has made an action request:
            "{user_message}"
            
            Based on this, determine what specific financial action they want to take:
            1. Set a budget
            2. Create a savings plan
            3. Check budget progress
            4. Detect unusual transactions
            5. Get investing advice
            6. Other
            
            Return ONLY the action number (1-6) without explanation.
            """
            
            try:
                action_response = model.generate_content(action_prompt)
                action = action_response.text.strip()
            except:
                action = "6"  # Default to other
                
            # Placeholder for action handling
            action_prompt = f"""
            The user wants to take a financial action:
            "{user_message}"
            
            Based on this request, craft a response that:
            1. Acknowledges what they want to do
            2. Explains what functionality is available to help them
            3. Guides them on how to use the appropriate features in the app
            4. Provides a helpful suggestion related to their goal
            """
            
            action_response = model.generate_content(action_prompt)
            return action_response.text.strip()
        else:  # General chat
            chat_prompt = f"""
            You are a helpful financial assistant AI agent. Respond to the user's message in a conversational way.
            
            Recent conversation:
            {formatted_chat}
            
            {financial_context}
            
            Keep your response helpful, conversational, and focused on financial topics.
            If the user asks about something unrelated to finances, gently bring the conversation back to financial topics.
            """
            
            response = model.generate_content(chat_prompt)
            chat_response = response.text.strip()
            
            # Add response to history
            self.add_to_chat_history('agent', chat_response)
            
            return chat_response
