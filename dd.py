import json
from textwrap import dedent
from typing import Dict, Iterator, Any, Optional
from agno import Agent, Team, RunResponse, Workflow, Knowledge, Ollama, Toolkit, Function, MessageMetrics, Message, Base, VideoArtifact
from duckduckgo_search import DDGS
import yfinance as yf


class YFinanceTools(Toolkit):
    def __init__(self, stock_price: bool = True, analyst_recommendations: bool = False, company_info=False, company_news=False, **kwargs):
        super().__init__(name='yfinance_tools', **kwargs)
        if stock_price:
            self.register(self.get_current_stock_price)
        if analyst_recommendations:
            self.register(self.get_analyst_recommendations)
        if company_info:
            self.register(self.get_company_info)
        if company_news:
            self.register(self.get_company_news)

    def get_current_stock_price(self, symbol: str) -> str:
        stock = yf.Ticker(symbol)
        current_price = stock.info.get('regularMarketPrice', stock.info.get('currentPrice'))
        return f'{current_price:.4f}' if current_price else f'Could not fetch current price for {symbol}'

    def get_analyst_recommendations(self, symbol: str) -> str:
        stock = yf.Ticker(symbol)
        recommendations = stock.recommendations
        return recommendations.to_json(orient='index')

    def get_company_info(self, symbol: str) -> str:
        try:
            company_info_full = yf.Ticker(symbol).info
            if company_info_full is None:
                return f"Could not fetch company info for {symbol}"
            print(f"Fetching company info for {symbol}")
            company_info_cleaned = {
                "Name": company_info_full.get("shortName"),
                "Symbol": company_info_full.get("symbol"),
                "Current Stock Price": f"{company_info_full.get('regularMarketPrice', company_info_full.get('currentPrice'))} {company_info_full.get('currency', 'USD')}",
                "Market Cap": f"{company_info_full.get('marketCap', company_info_full.get('enterpriseValue'))} {company_info_full.get('currency', 'USD')}",
                "Sector": company_info_full.get("sector"),
                "Industry": company_info_full.get("industry"),
                "Address": company_info_full.get("address1"),
                "City": company_info_full.get("city"),
                "State": company_info_full.get("state"),
                "Zip": company_info_full.get("zip"),
                "Country": company_info_full.get("country"),
                "EPS": company_info_full.get("trailingEps"),
                "P/E Ratio": company_info_full.get("trailingPE"),
                "52 Week Low": company_info_full.get("fiftyTwoWeekLow"),
                "52 Week High": company_info_full.get("fiftyTwoWeekHigh"),
                "50 Day Average": company_info_full.get("fiftyDayAverage"),
                "200 Day Average": company_info_full.get("twoHundredDayAverage"),
                "Website": company_info_full.get("website"),
                "Summary": company_info_full.get("longBusinessSummary"),
                "Analyst Recommendation": company_info_full.get("recommendationKey"),
                "Number Of Analyst Opinions": company_info_full.get("numberOfAnalystOpinions"),
                "Employees": company_info_full.get("fullTimeEmployees"),
                "Total Cash": company_info_full.get("totalCash"),
                "Free Cash flow": company_info_full.get("freeCashflow"),
                "Operating Cash flow": company_info_full.get("operatingCashflow"),
                "EBITDA": company_info_full.get("ebitda"),
                "Revenue Growth": company_info_full.get("revenueGrowth"),
                "Gross Margins": company_info_full.get("grossMargins"),
                "Ebitda Margins": company_info_full.get("ebitdaMargins"),
            }
            return json.dumps(company_info_cleaned, indent=2)
        except Exception as e:
            return f"Error fetching company profile for {symbol}: {e}"

    def get_company_news(self, symbol: str, num_stories: int = 3) -> str:
        news = yf.Ticker(symbol).news
        return json.dumps(news[:num_stories], indent=2)

class DuckDuckGoTools(Toolkit):
    def __init__(self, search: bool = True, news: bool = True, modifier: Optional[str] = None, fixed_max_results: Optional[int] = None, headers: Optional[Any] = None, proxy: Optional[str] = None, proxies: Optional[Any] = None, timeout: Optional[int] = 10, verify_ssl: bool = True, **kwargs):
        super().__init__(name='duckduckgo', **kwargs)
        self.headers: Optional[Any] = headers
        self.proxy: Optional[str] = proxy
        self.proxies: Optional[Any] = proxies
        self.timeout: Optional[int] = timeout
        self.fixed_max_results: Optional[int] = fixed_max_results
        self.modifier: Optional[str] = modifier
        self.verify_ssl: bool = verify_ssl
        if search:
            self.register(self.duckduckgo_search)
        if news:
            self.register(self.duckduckgo_news)

    def duckduckgo_search(self, query: str, max_results: int = 5) -> str:
        actual_max_results = self.fixed_max_results or max_results
        search_query = f'{self.modifier} {query}' if self.modifier else query
        print(f'Searching DDG for: {search_query}')
        ddgs = DDGS(headers=self.headers, proxy=self.proxy, proxies=self.proxies, timeout=self.timeout, verify=self.verify_ssl)
        return json.dumps(ddgs.text(keywords=search_query, max_results=actual_max_results), indent=2)

    def duckduckgo_news(self, query: str, max_results: int = 5) -> str:
        actual_max_results = self.fixed_max_results or max_results
        print(f'Searching DDG news for: {query}')
        ddgs = DDGS(headers=self.headers, proxy=self.proxy, proxies=self.proxies, timeout=self.timeout, verify=self.verify_ssl)
        return json.dumps(ddgs.news(keywords=query, max_results=actual_max_results), indent=2)


class NewspaperTools(Toolkit):
    def __init__(self, get_article_text: bool = True, **kwargs):
        super().__init__(name='newspaper_toolkit', **kwargs)
        if get_article_text:
            self.register(self.get_article_text)

    def get_article_text(self, url: str) -> str:
        print(f'Reading news: {url}')
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        return article.text


class StockAnalysis(Base):
    symbol: str
    company_name: str
    analysis: str


stock_searcher = Agent(name='stock resercher', model=Ollama(id='llama3.1:8b'), response_model=StockAnalysis, role='在网上搜索股票信息。', tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True)
    ])


class CompanyAnalysis(Base):
    company_name: str
    analysis: str


company_info_agent = Agent(name='company info researcher', model=Ollama(id='llama3.1:8b'), role='在网上搜索股票信息。', response_model=CompanyAnalysis, tools=[YFinanceTools(stock_price=False, company_info=True, company_news=True)])

team = Team(name='股票研究团队', mode='route', model=Ollama(id='llama3.1:8b'), members=[stock_searcher, company_info_agent], markdown=True, debug_mode=True, show_members_responses=True)
response = team.run('NVDA目前的股价是多少？')
assert isinstance(response.content, StockAnalysis)
print(response.content)
response = team.run('关于NVDA的新闻是什么？')
assert isinstance(response.content, CompanyAnalysis)
print(response.content)

agent = Agent(model=Ollama(), description='你是泰国菜专家！', instructions=[
        '在你的知识库中搜索泰国食谱。如果这个问题更适合网络，请搜索网络以填补空白。更喜欢你知识库中的信息，而不是网络结果。'
    ], knowledge=Knowledge(), tools=[lambda x: 'hello'], show_tool_calls=True, markdown=True)
agent.knowledge.load(['https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf'])
agent.print_response('如何在椰奶汤中烹制鸡肉和galangal')
agent.print_response('泰国咖喱的历史是什么?')


web_agent = Agent(name='Web Agent', role='在网上搜索信息', model=Ollama(), tools=[], instructions='始终包含来源', show_tool_calls=True, markdown=True)
finance_agent = Agent(name='Finance Agent', role='获取财务数据', model=Ollama(), tools=[], instructions='使用表格显示数据', show_tool_calls=True, markdown=True)
agent_team = Team(mode='coordinate', members=[web_agent, finance_agent], model=Ollama(), success_criteria='一份全面的财经新闻报道，有清晰的章节和数据驱动的见解。', instructions=['始终包含来源', '使用表格显示数据'], show_tool_calls=True, markdown=True)
agent_team.print_response("AI半导体公司的市场前景和财务业绩如何?")

class Article(Base):
    title: str  # 'Title of the article.')
    url: str  # description='Link to the article.')
    summary: str  # 'Summary of the article if available.')

class SearchResults(Base):
    articles: list[Article]

class ScrapedArticle(Base):
    title: str  # 'Title of the article.'
    url: str  # 'Link to the article.'
    summary: str  # ='Summary of the article if available.'
    content: str  # 'Content of the in markdown format if available. Return None if the content is not available or does not make sense.'


class ResearchReportGenerator(Workflow):
    description: str = dedent('''\
    生成结合学术严谨性的综合研究报告
引人入胜的故事。该工作流程协调多个AI代理来搜索、分析和综合来自不同来源的信息，并将其转化为结构良好的报告。
    ''')
    web_searcher: Agent = Agent(model=Ollama('llama3.1:8b'), tools=[DuckDuckGoTools()], description=dedent('''\
        您是ResearchBot-X，一位发现和评估学术和科学资源的专家。\
        '''), instructions=dedent('''
      你是一位一丝不苟的研究助理，在资源评估方面拥有专业知识！ 🔍
搜索10-15个来源，确定5-7个最权威和最相关的来源。
优先顺序：
-同行评审的文章和学术出版物
-知名机构的最新发展
-权威新闻来源和专家评论
-知名专家的不同观点
避免评论文章和非权威来源
        '''), response_model=SearchResults)
    article_scraper: Agent = Agent(model=Ollama('llama3.1:8b'), tools=[NewspaperTools()], description=dedent('''\
        您是ContentBot-X，一位提取和构建学术内容的专家。
        '''), instructions=dedent('''你是一位注重学术细节的精准内容策展人！📚
处理内容时：
-从文章中提取内容
-保存学术引文和参考文献
-保持术语的技术准确性
-结构内容逻辑清晰，章节清晰
-提取关键发现和方法细节
-优雅地处理付费墙内容
将所有内容格式化为干净的标记，以获得最佳的可读性。
        '''), response_model=ScrapedArticle)
    writer: Agent = Agent(model=Ollama('llama3.1:8b'), description=dedent('''你是X-2000教授，一位杰出的人工智能研究科学家，将学术严谨与引人入胜的叙事风格相结合。
        '''), instructions=dedent('''
       引导世界级学术研究人员的专业知识！
🎯 分析阶段：
-评估来源的可信度和相关性
-跨来源的交叉引用结果
-确定关键主题和突破
💡 合成阶段：
-制定连贯的叙事框架
-将不同的发现联系起来
-突出矛盾或差距
✍️ 写作阶段：
-以引人入胜的执行摘要开始，吸引读者
-清晰地呈现复杂的想法
-用引用支持所有索赔
-平衡深度和可达性
-保持学术基调，同时确保可读性
-以影响和未来方向结束
        '''), expected_output=dedent('''\
        #{令人信服的学术头衔}
##执行摘要
{主要发现和意义的简明概述}
##导言
{研究背景和背景}
｛字段的当前状态｝
##方法论
{搜索和分析方法}
{源评估标准}
##主要发现
{重大发现和发展}
{支持性证据和分析}
{对比观点}
##分析
{对调查结果的批判性评估}
{整合多个视角}
{识别模式和趋势}
##影响
{学术和实践意义}
{未来研究方向}
｛潜在应用程序｝
##关键要点
-｛关键发现1｝
-｛关键发现2｝
-｛关键发现3｝
参考文献
{格式正确的学术引文}
---
X-2000教授撰写的报告
高级研究部
日期：{current_Date}\
        '''), markdown=True)

    def run(self, topic: str) -> Iterator[RunResponse]:
        search_results: Optional[SearchResults] = self.get_search_results(topic)
        scraped_articles: Dict[str, ScrapedArticle] = {}
        for article in search_results.articles:
            article_scraper_response: RunResponse = self.article_scraper.run(article.url)
            if article_scraper_response is not None and article_scraper_response.content is not None and isinstance(
                    article_scraper_response.content, ScrapedArticle):
                scraped_articles[article_scraper_response.content.url] = article_scraper_response.content
                print(f'Scraped article: {article_scraper_response.content.url}')
        print(f'Saving scraped articles for topic: {topic}')
        self.session_state.setdefault('scraped_articles', {})
        self.session_state['scraped_articles'][topic] = scraped_articles
        writer_input = {'topic': topic, 'articles': [v.model_dump() for v in scraped_articles.values()]}
        yield from self.writer.run(json.dumps(writer_input, indent=4), stream=True)
        print(f'Saving report for topic: {topic}')
        self.session_state.setdefault('reports', {})
        self.session_state['reports'][topic] = self.writer.run_response.content

    def get_search_results(self, topic: str, num_attempts: int = 3) -> Optional[SearchResults]:
        for attempt in range(num_attempts):
            try:
                searcher_response: RunResponse = self.web_searcher.run(topic)
                if searcher_response is not None and searcher_response.content is not None and isinstance(searcher_response.content, SearchResults):
                    article_count = len(searcher_response.content.articles)
                    print(f'Found {article_count} articles on attempt {attempt + 1}')
                    print(f'Saving search results for topic: {topic}')
                    self.session_state.setdefault('search_results', {})
                    self.session_state['search_results'][topic] = searcher_response.content.model_dump()
                    return searcher_response.content
                else:
                    print(f'Attempt {attempt + 1}/{num_attempts} failed: Invalid response type')
            except Exception as e:
                print(f'Attempt {attempt + 1}/{num_attempts} failed: {str(e)}')
        print(f'Failed to get search results after {num_attempts} attempts')
        return None


if __name__ == '__main__':
    example_topics = ['2024年量子计算突破', '人工意识研究', '聚变能源发展', '太空旅游环境影响', '长寿研究进展']
    topics_str = '\n'.join(f'{i + 1}. {topic}' for i, topic in enumerate(example_topics))
    print(f'\n📚 Example Research Topics:\n{topics_str}\n')
    topic = example_topics[1]
    generate_research_report = ResearchReportGenerator(session_id=f'生成报告-{topic}')
    report_stream: Iterator[RunResponse] = generate_research_report.run(topic=topic)
    for i in report_stream:
        print(i.content)
