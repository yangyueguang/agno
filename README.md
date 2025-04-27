<div align='center' id='top'>
  <a href='https://docs.agno.com'>
    <picture>
      <source media='(prefers-color-scheme: dark)' srcset='https://agno-public.s3.us-east-1.amazonaws.com/assets/logo-dark.svg'>
      <source media='(prefers-color-scheme: light)' srcset='https://agno-public.s3.us-east-1.amazonaws.com/assets/logo-light.svg'>
      <img src='https://agno-public.s3.us-east-1.amazonaws.com/assets/logo-light.svg' alt='Agno'>
    </picture>
  </a>
</div>
<div align='center'>
  <a href='https://docs.agno.com'>📚 Documentation</a> &nbsp;|&nbsp;
  <a href='https://docs.agno.com/examples/introduction'>💡 Examples</a> &nbsp;|&nbsp;
  <a href='https://github.com/agno-agi/agno/stargazers'>🌟 Star Us</a>
</div>

## 说明
[Agno](https://docs.agno.com) 简化板


```python
from agno.agent import Agent
from agno.ollama import Ollama
from agno.knowledge import PDFUrlKnowledgeBase
agent = Agent(model=Ollama(), description='You are a Thai cuisine expert!', instructions=[
        'Search your knowledge base for Thai recipes.', 'If the question is better suited for the web, search the web to fill in gaps.', 'Prefer the information in your knowledge base over the web results.'
    ], knowledge=PDFUrlKnowledgeBase(urls=['https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf']), tools=[lambda x: 'hello'], show_tool_calls=True, markdown=True)
agent.knowledge.load()
agent.print_response('How do I make chicken and galangal in coconut milk soup', stream=True)
agent.print_response('What is the history of Thai curry?', stream=True)
```
```python
from agno.agent import Agent
from agno.ollama import Ollama
from agno.team import Team
web_agent = Agent(name='Web Agent', role='Search the web for information', model=Ollama(), tools=[], instructions='Always include sources', show_tool_calls=True, markdown=True)
finance_agent = Agent(name='Finance Agent', role='Get financial data', model=Ollama(), tools=[], instructions='Use tables to display data', show_tool_calls=True, markdown=True)
agent_team = Team(mode='coordinate', members=[web_agent, finance_agent], model=Ollama(), success_criteria='A comprehensive financial news report with clear sections and data-driven insights.', instructions=['Always include sources', 'Use tables to display data'], show_tool_calls=True, markdown=True)
agent_team.print_response("What's the market outlook and financial performance of AI semiconductor companies?", stream=True)
```
