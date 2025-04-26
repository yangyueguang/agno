'''高级研究工作流程-您的人工智能研究助理！
本示例展示了如何构建一个复杂的研究工作流程，该流程结合了以下内容：
🔍 用于查找相关来源的网络搜索功能
📚 内容提取和处理
✍️ 学术风格报告生成
💾 智能缓存可提高性能
我们使用了以下免费工具：
-DuckDuckGoTools：在网络上搜索相关文章
-Newspaper4kTools：废料和处理文章内容
要尝试的示例研究主题：
-“量子计算的最新进展是什么？”
-“研究人工意识的现状”
-“分析聚变能源的最新突破”
-“调查太空旅游的环境影响”
-“探索长寿研究的最新发现”
'''
import json
from textwrap import dedent
from typing import Dict, Iterator, Optional, Union, Any, Optional, Iterable
from agno.storage import SqliteStorage
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.ollama import Ollama
from agno.team import Team
from agno.run import RunResponse
from agno.models import Timer

from newspaper import Article
from duckduckgo_search import DDGS
import yfinance as yf
from agno.tools import Toolkit


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
        article = Article(url)
        article.download()
        article.parse()
        return article.text


def pprint_run_response(run_response: Union[RunResponse, Iterable[RunResponse]], markdown: bool = False, show_time: bool = False) -> None:
    from rich.box import ROUNDED
    from rich.json import JSON
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.status import Status
    from rich.table import Table
    from rich.console import Console
    console = Console()
    if isinstance(run_response, RunResponse):
        single_response_content: Union[str, JSON, Markdown] = ''
        if isinstance(run_response.content, str):
            single_response_content = (Markdown(run_response.content) if markdown else run_response.get_content_as_string(indent=4))
        elif isinstance(run_response.content, BaseModel):
            try:
                single_response_content = JSON(run_response.content.model_dump_json(exclude_none=True), indent=2)
            except Exception as e:
                print(f'Failed to convert response to Markdown: {e}')
        else:
            try:
                single_response_content = JSON(json.dumps(run_response.content), indent=4)
            except Exception as e:
                print(f'Failed to convert response to string: {e}')
        table = Table(box=ROUNDED, border_style='blue', show_header=False)
        table.add_row(single_response_content)
        console.print(table)
    else:
        streaming_response_content: str = ''
        with Live(console=console) as live_log:
            status = Status('Working...', spinner='dots')
            live_log.update(status)
            response_timer = Timer()
            response_timer.start()
            for resp in run_response:
                if isinstance(resp, RunResponse) and isinstance(resp.content, str):
                    streaming_response_content += resp.content
                formatted_response = Markdown(streaming_response_content) if markdown else streaming_response_content
                table = Table(box=ROUNDED, border_style='blue', show_header=False)
                if show_time:
                    table.add_row(f'Response\n({response_timer.elapsed:.1f}s)', formatted_response)
                else:
                    table.add_row(formatted_response)
                live_log.update(table)
            response_timer.stop()


class StockAnalysis(BaseModel):
    symbol: str
    company_name: str
    analysis: str


stock_searcher = Agent(name='stock resercher', model=Ollama('llama3.1:8b'), response_model=StockAnalysis, role='在网上搜索股票信息。', tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True)
    ])


class CompanyAnalysis(BaseModel):
    company_name: str
    analysis: str


company_info_agent = Agent(name='company info researcher', model=Ollama('llama3.1:8b'), role='在网上搜索股票信息。', response_model=CompanyAnalysis, tools=[YFinanceTools(stock_price=False, company_info=True, company_news=True)])

team = Team(name='股票研究团队', mode='route', model=Ollama('llama3.1:8b'), members=[stock_searcher, company_info_agent], markdown=True, debug_mode=True, show_members_responses=True)
response = team.run('NVDA目前的股价是多少？')
assert isinstance(response.content, StockAnalysis)
print(response.content)
response = team.run('关于NVDA的新闻是什么？')
assert isinstance(response.content, CompanyAnalysis)
print(response.content)

class Article(BaseModel):
    title: str = Field(..., description='Title of the article.')
    url: str = Field(..., description='Link to the article.')
    summary: Optional[str] = Field(..., description='Summary of the article if available.')

class SearchResults(BaseModel):
    articles: list[Article]

class ScrapedArticle(BaseModel):
    title: str = Field(..., description='Title of the article.')
    url: str = Field(..., description='Link to the article.')
    summary: Optional[str] = Field(..., description='Summary of the article if available.')
    content: Optional[str] = Field(..., description='Content of the in markdown format if available. Return None if the content is not available or does not make sense.')


class ResearchReportGenerator(Workflow):
    description: str = dedent('''\
    Generate comprehensive research reports that combine academic rigor
    with engaging storytelling. This workflow orchestrates multiple AI agents to search, analyze, and synthesize information from diverse sources into well-structured reports.
    ''')
    web_searcher: Agent = Agent(model=Ollama('llama3.1:8b'), tools=[DuckDuckGoTools()], description=dedent('''\
        You are ResearchBot-X, an expert at discovering and evaluating academic and scientific sources.\
        '''), instructions=dedent('''\
        You're a meticulous research assistant with expertise in source evaluation! 🔍
        Search for 10-15 sources and identify the 5-7 most authoritative and relevant ones.
        Prioritize:
        - Peer-reviewed articles and academic publications
        - Recent developments from reputable institutions
        - Authoritative news sources and expert commentary
        - Diverse perspectives from recognized experts
        Avoid opinion pieces and non-authoritative sources.\
        '''), response_model=SearchResults)
    article_scraper: Agent = Agent(model=Ollama('llama3.1:8b'), tools=[NewspaperTools()], description=dedent('''\
        You are ContentBot-X, an expert at extracting and structuring academic content.\
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
    writer: Agent = Agent(model=Ollama('llama3.1:8b'), description=dedent('''\
        You are Professor X-2000, a distinguished AI research scientist combining academic rigor with engaging narrative style.\
        '''), instructions=dedent('''\
        Channel the expertise of a world-class academic researcher!
        🎯 Analysis Phase:
          - Evaluate source credibility and relevance
          - Cross-reference findings across sources
          - Identify key themes and breakthroughs
        💡 Synthesis Phase:
          - Develop a coherent narrative framework
          - Connect disparate findings
          - Highlight contradictions or gaps
        ✍️ Writing Phase:
          - Begin with an engaging executive summary, hook the reader
          - Present complex ideas clearly
          - Support all claims with citations
          - Balance depth with accessibility
          - Maintain academic tone while ensuring readability
          - End with implications and future directions\
        '''), expected_output=dedent('''\
        # {Compelling Academic Title}
        ## Executive Summary
        {Concise overview of key findings and significance}
        ## Introduction
        {Research context and background}
        {Current state of the field}
        ## Methodology
        {Search and analysis approach}
        {Source evaluation criteria}
        ## Key Findings
        {Major discoveries and developments}
        {Supporting evidence and analysis}
        {Contrasting viewpoints}
        ## Analysis
        {Critical evaluation of findings}
        {Integration of multiple perspectives}
        {Identification of patterns and trends}
        ## Implications
        {Academic and practical significance}
        {Future research directions}
        {Potential applications}
        ## Key Takeaways
        - {Critical finding 1}
        - {Critical finding 2}
        - {Critical finding 3}
        ## References
        {Properly formatted academic citations}
        ---
        Report generated by Professor X-2000
        Advanced Research Division
        Date: {current_date}\
        '''), markdown=True)

    def run(self, topic: str, use_search_cache: bool = True, use_scrape_cache: bool = True, use_cached_report: bool = True) -> Iterator[RunResponse]:
        print(f'Generating a report on: {topic}')
        if use_cached_report:
            cached_report = self.get_cached_report(topic)
            if cached_report:
                yield RunResponse(content=cached_report, event=RunEvent.workflow_completed)
                return
        search_results: Optional[SearchResults] = self.get_search_results(topic, use_search_cache)
        if search_results is None or len(search_results.articles) == 0:
            yield RunResponse(event=RunEvent.workflow_completed, content=f'Sorry, could not find any articles on the topic: {topic}')
            return
        scraped_articles: Dict[str, ScrapedArticle] = self.scrape_articles(search_results, use_scrape_cache)
        yield from self.write_research_report(topic, scraped_articles)

    def get_cached_report(self, topic: str) -> Optional[str]:
        print('Checking if cached report exists')
        return self.session_state.get('reports', {}).get(topic)

    def add_report_to_cache(self, topic: str, report: str):
        print(f'Saving report for topic: {topic}')
        self.session_state.setdefault('reports', {})
        self.session_state['reports'][topic] = report
        self.write_to_storage()

    def get_cached_search_results(self, topic: str) -> Optional[SearchResults]:
        print('Checking if cached search results exist')
        return self.session_state.get('search_results', {}).get(topic)

    def add_search_results_to_cache(self, topic: str, search_results: SearchResults):
        print(f'Saving search results for topic: {topic}')
        self.session_state.setdefault('search_results', {})
        self.session_state['search_results'][topic] = search_results.model_dump()
        self.write_to_storage()

    def get_cached_scraped_articles(self, topic: str) -> Optional[Dict[str, ScrapedArticle]]:
        print('Checking if cached scraped articles exist')
        return self.session_state.get('scraped_articles', {}).get(topic)

    def add_scraped_articles_to_cache(self, topic: str, scraped_articles: Dict[str, ScrapedArticle]):
        print(f'Saving scraped articles for topic: {topic}')
        self.session_state.setdefault('scraped_articles', {})
        self.session_state['scraped_articles'][topic] = scraped_articles
        self.write_to_storage()

    def get_search_results(self, topic: str, use_search_cache: bool, num_attempts: int = 3) -> Optional[SearchResults]:
        if use_search_cache:
            try:
                search_results_from_cache = self.get_cached_search_results(topic)
                if search_results_from_cache is not None:
                    search_results = SearchResults.model_validate(search_results_from_cache)
                    print(f'Found {len(search_results.articles)} articles in cache.')
                    return search_results
            except Exception as e:
                print(f'Could not read search results from cache: {e}')
        for attempt in range(num_attempts):
            try:
                searcher_response: RunResponse = self.web_searcher.run(topic)
                if searcher_response is not None and searcher_response.content is not None and isinstance(searcher_response.content, SearchResults):
                    article_count = len(searcher_response.content.articles)
                    print(f'Found {article_count} articles on attempt {attempt + 1}')
                    self.add_search_results_to_cache(topic, searcher_response.content)
                    return searcher_response.content
                else:
                    print(f'Attempt {attempt + 1}/{num_attempts} failed: Invalid response type')
            except Exception as e:
                print(f'Attempt {attempt + 1}/{num_attempts} failed: {str(e)}')
        print(f'Failed to get search results after {num_attempts} attempts')
        return None

    def scrape_articles(self, search_results: SearchResults, use_scrape_cache: bool) -> Dict[str, ScrapedArticle]:
        scraped_articles: Dict[str, ScrapedArticle] = {}
        if use_scrape_cache:
            try:
                scraped_articles_from_cache = self.get_cached_scraped_articles(topic)
                if scraped_articles_from_cache is not None:
                    scraped_articles = scraped_articles_from_cache
                    print(f'Found {len(scraped_articles)} scraped articles in cache.')
                    return scraped_articles
            except Exception as e:
                print(f'Could not read scraped articles from cache: {e}')
        for article in search_results.articles:
            if article.url in scraped_articles:
                print(f'Found scraped article in cache: {article.url}')
                continue
            article_scraper_response: RunResponse = self.article_scraper.run(article.url)
            if article_scraper_response is not None and article_scraper_response.content is not None and isinstance(article_scraper_response.content, ScrapedArticle):
                scraped_articles[article_scraper_response.content.url] = (article_scraper_response.content)
                print(f'Scraped article: {article_scraper_response.content.url}')
        self.add_scraped_articles_to_cache(topic, scraped_articles)
        return scraped_articles

    def write_research_report(self, topic: str, scraped_articles: Dict[str, ScrapedArticle]) -> Iterator[RunResponse]:
        print('Writing research report')
        writer_input = {'topic': topic, 'articles': [v.model_dump() for v in scraped_articles.values()]}
        yield from self.writer.run(json.dumps(writer_input, indent=4), stream=True)
        self.add_report_to_cache(topic, self.writer.run_response.content)


if __name__ == '__main__':
    example_topics = [
        'quantum computing breakthroughs 2024', 'artificial consciousness research', 'fusion energy developments', 'space tourism environmental impact', 'longevity research advances', ]
    topics_str = '\n'.join(f'{i + 1}. {topic}' for i, topic in enumerate(example_topics))
    print(f'\n📚 Example Research Topics:\n{topics_str}\n')
    topic = 'quantum computing breakthroughs 2024'
    url_safe_topic = topic.lower().replace(' ', '-')
    generate_research_report = ResearchReportGenerator(session_id=f'generate-report-on-{url_safe_topic}', storage=SqliteStorage(table_name='generate_research_report_workflow', db_file='workflows.db'))
    report_stream: Iterator[RunResponse] = generate_research_report.run(topic=topic, use_search_cache=True, use_scrape_cache=True, use_cached_report=True)
    pprint_run_response(report_stream, markdown=True)
