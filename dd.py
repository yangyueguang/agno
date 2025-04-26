"""高级研究工作流程-您的人工智能研究助理！
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
"""
import json
from textwrap import dedent
from typing import Dict, Iterator, Optional
from agno.storage import SqliteStorage
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field
from typing import Iterator
from pydantic import BaseModel
from agno.agent import Agent
from agno.ollama import Ollama
from agno.team import Team
import json
from typing import Iterable, Union
from pydantic import BaseModel
from agno.run import RunResponse
from agno.models import Timer

  ########################## 下面要删除的
try:
    from newspaper import Article
except ImportError:
    raise ImportError("`newspaper3k` not installed. Please run `pip install newspaper3k lxml_html_clean`.")
import json
from typing import Any, Optional
from agno.tools import Toolkit

try:
    from duckduckgo_search import DDGS
except ImportError:
    raise ImportError("`duckduckgo-search` not installed. Please install using `pip install duckduckgo-search`")
import json
from agno.tools import Toolkit

try:
    import yfinance as yf
except ImportError:
    raise ImportError("`yfinance` not installed. Please install using `pip install yfinance`.")

class YFinanceTools(Toolkit):
    """
    YFinanceTools is a toolkit for getting financial data from Yahoo Finance.
    Args:
        stock_price (bool): Whether to get the current stock price.
        company_info (bool): Whether to get company information.
        stock_fundamentals (bool): Whether to get stock fundamentals.
        income_statements (bool): Whether to get income statements.
        key_financial_ratios (bool): Whether to get key financial ratios.
        analyst_recommendations (bool): Whether to get analyst recommendations.
        company_news (bool): Whether to get company news.
        technical_indicators (bool): Whether to get technical indicators.
        historical_prices (bool): Whether to get historical prices.
        enable_all (bool): Whether to enable all tools.
    """

    def __init__(self,
        stock_price: bool = True,
        company_info: bool = False,
        stock_fundamentals: bool = False,
        income_statements: bool = False,
        key_financial_ratios: bool = False,
        analyst_recommendations: bool = False,
        company_news: bool = False,
        technical_indicators: bool = False,
        historical_prices: bool = False,
        enable_all: bool = False,
        **kwargs):
        super().__init__(name="yfinance_tools", **kwargs)
        if stock_price or enable_all:
            self.register(self.get_current_stock_price)
        if company_info or enable_all:
            self.register(self.get_company_info)
        if stock_fundamentals or enable_all:
            self.register(self.get_stock_fundamentals)
        if income_statements or enable_all:
            self.register(self.get_income_statements)
        if key_financial_ratios or enable_all:
            self.register(self.get_key_financial_ratios)
        if analyst_recommendations or enable_all:
            self.register(self.get_analyst_recommendations)
        if company_news or enable_all:
            self.register(self.get_company_news)
        if technical_indicators or enable_all:
            self.register(self.get_technical_indicators)
        if historical_prices or enable_all:
            self.register(self.get_historical_stock_prices)

    def get_current_stock_price(self, symbol: str) -> str:
        """
        Use this function to get the current stock price for a given symbol.
        Args:
            symbol (str): The stock symbol.
        Returns:
            str: The current stock price or error message.
        """
        try:
            print(f"Fetching current price for {symbol}")
            stock = yf.Ticker(symbol)
            # Use "regularMarketPrice" for regular market hours, or "currentPrice" for pre/post market
            current_price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
            return f"{current_price:.4f}" if current_price else f"Could not fetch current price for {symbol}"
        except Exception as e:
            return f"Error fetching current price for {symbol}: {e}"

    def get_company_info(self, symbol: str) -> str:
        """Use this function to get company information and overview for a given stock symbol.
        Args:
            symbol (str): The stock symbol.
        Returns:
            str: JSON containing company profile and overview.
        """
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

    def get_historical_stock_prices(self, symbol: str, period: str = "1mo", interval: str = "1d") -> str:
        """
        Use this function to get the historical stock price for a given symbol.
        Args:
            symbol (str): The stock symbol.
            period (str): The period for which to retrieve historical prices. Defaults to "1mo".
                        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval (str): The interval between data points. Defaults to "1d".
                        Valid intervals: 1d,5d,1wk,1mo,3mo
        Returns:
          str: The current stock price or error message.
        """
        try:
            print(f"Fetching historical prices for {symbol}")
            stock = yf.Ticker(symbol)
            historical_price = stock.history(period=period, interval=interval)
            return historical_price.to_json(orient="index")
        except Exception as e:
            return f"Error fetching historical prices for {symbol}: {e}"

    def get_stock_fundamentals(self, symbol: str) -> str:
        """Use this function to get fundamental data for a given stock symbol yfinance API.
        Args:
            symbol (str): The stock symbol.
        Returns:
            str: A JSON string containing fundamental data or an error message.
                Keys:
                    - 'symbol': The stock symbol.
                    - 'company_name': The long name of the company.
                    - 'sector': The sector to which the company belongs.
                    - 'industry': The industry to which the company belongs.
                    - 'market_cap': The market capitalization of the company.
                    - 'pe_ratio': The forward price-to-earnings ratio.
                    - 'pb_ratio': The price-to-book ratio.
                    - 'dividend_yield': The dividend yield.
                    - 'eps': The trailing earnings per share.
                    - 'beta': The beta value of the stock.
                    - '52_week_high': The 52-week high price of the stock.
                    - '52_week_low': The 52-week low price of the stock.
        """
        try:
            print(f"Fetching fundamentals for {symbol}")
            stock = yf.Ticker(symbol)
            info = stock.info
            fundamentals = {
                "symbol": symbol,
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("forwardPE", "N/A"),
                "pb_ratio": info.get("priceToBook", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "eps": info.get("trailingEps", "N/A"),
                "beta": info.get("beta", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            }
            return json.dumps(fundamentals, indent=2)
        except Exception as e:
            return f"Error getting fundamentals for {symbol}: {e}"

    def get_income_statements(self, symbol: str) -> str:
        """Use this function to get income statements for a given stock symbol.
        Args:
            symbol (str): The stock symbol.
        Returns:
            dict: JSON containing income statements or an empty dictionary.
        """
        try:
            print(f"Fetching income statements for {symbol}")
            stock = yf.Ticker(symbol)
            financials = stock.financials
            return financials.to_json(orient="index")
        except Exception as e:
            return f"Error fetching income statements for {symbol}: {e}"

    def get_key_financial_ratios(self, symbol: str) -> str:
        """Use this function to get key financial ratios for a given stock symbol.
        Args:
            symbol (str): The stock symbol.
        Returns:
            dict: JSON containing key financial ratios.
        """
        try:
            print(f"Fetching key financial ratios for {symbol}")
            stock = yf.Ticker(symbol)
            key_ratios = stock.info
            return json.dumps(key_ratios, indent=2)
        except Exception as e:
            return f"Error fetching key financial ratios for {symbol}: {e}"

    def get_analyst_recommendations(self, symbol: str) -> str:
        """Use this function to get analyst recommendations for a given stock symbol.
        Args:
            symbol (str): The stock symbol.
        Returns:
            str: JSON containing analyst recommendations.
        """
        try:
            print(f"Fetching analyst recommendations for {symbol}")
            stock = yf.Ticker(symbol)
            recommendations = stock.recommendations
            return recommendations.to_json(orient="index")
        except Exception as e:
            return f"Error fetching analyst recommendations for {symbol}: {e}"

    def get_company_news(self, symbol: str, num_stories: int = 3) -> str:
        """Use this function to get company news and press releases for a given stock symbol.
        Args:
            symbol (str): The stock symbol.
            num_stories (int): The number of news stories to return. Defaults to 3.
        Returns:
            str: JSON containing company news and press releases.
        """
        try:
            print(f"Fetching company news for {symbol}")
            news = yf.Ticker(symbol).news
            return json.dumps(news[:num_stories], indent=2)
        except Exception as e:
            return f"Error fetching company news for {symbol}: {e}"

    def get_technical_indicators(self, symbol: str, period: str = "3mo") -> str:
        """Use this function to get technical indicators for a given stock symbol.
        Args:
            symbol (str): The stock symbol.
            period (str): The time period for which to retrieve technical indicators.
                Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. Defaults to 3mo.
        Returns:
            str: JSON containing technical indicators.
        """
        try:
            print(f"Fetching technical indicators for {symbol}")
            indicators = yf.Ticker(symbol).history(period=period)
            return indicators.to_json(orient="index")
        except Exception as e:
            return f"Error fetching technical indicators for {symbol}: {e}"

class DuckDuckGoTools(Toolkit):
    """
    DuckDuckGo is a toolkit for searching DuckDuckGo easily.
    Args:
        search (bool): Enable DuckDuckGo search function.
        news (bool): Enable DuckDuckGo news function.
        modifier (Optional[str]): A modifier to be used in the search request.
        fixed_max_results (Optional[int]): A fixed number of maximum results.
        headers (Optional[Any]): Headers to be used in the search request.
        proxy (Optional[str]): Proxy to be used in the search request.
        proxies (Optional[Any]): A list of proxies to be used in the search request.
        timeout (Optional[int]): The maximum number of seconds to wait for a response.
    """

    def __init__(self,
        search: bool = True,
        news: bool = True,
        modifier: Optional[str] = None,
        fixed_max_results: Optional[int] = None,
        headers: Optional[Any] = None,
        proxy: Optional[str] = None,
        proxies: Optional[Any] = None,
        timeout: Optional[int] = 10,
        verify_ssl: bool = True,
        **kwargs):
        super().__init__(name="duckduckgo", **kwargs)
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
        """Use this function to search DuckDuckGo for a query.
        Args:
            query(str): The query to search for.
            max_results (optional, default=5): The maximum number of results to return.
        Returns:
            The result from DuckDuckGo.
        """
        actual_max_results = self.fixed_max_results or max_results
        search_query = f"{self.modifier} {query}" if self.modifier else query
        print(f"Searching DDG for: {search_query}")
        ddgs = DDGS(headers=self.headers, proxy=self.proxy, proxies=self.proxies, timeout=self.timeout, verify=self.verify_ssl)
        return json.dumps(ddgs.text(keywords=search_query, max_results=actual_max_results), indent=2)

    def duckduckgo_news(self, query: str, max_results: int = 5) -> str:
        """Use this function to get the latest news from DuckDuckGo.
        Args:
            query(str): The query to search for.
            max_results (optional, default=5): The maximum number of results to return.
        Returns:
            The latest news from DuckDuckGo.
        """
        actual_max_results = self.fixed_max_results or max_results
        print(f"Searching DDG news for: {query}")
        ddgs = DDGS(headers=self.headers, proxy=self.proxy, proxies=self.proxies, timeout=self.timeout, verify=self.verify_ssl)
        return json.dumps(ddgs.news(keywords=query, max_results=actual_max_results), indent=2)
class NewspaperTools(Toolkit):
    """
    Newspaper is a tool for getting the text of an article from a URL.
    Args:
        get_article_text (bool): Whether to get the text of an article from a URL.
    """

    def __init__(self, get_article_text: bool = True, **kwargs):
        super().__init__(name="newspaper_toolkit", **kwargs)
        if get_article_text:
            self.register(self.get_article_text)

    def get_article_text(self, url: str) -> str:
        """Get the text of an article from a URL.
        Args:
            url (str): The URL of the article.
        Returns:
            str: The text of the article.
        """
        try:
            print(f"Reading news: {url}")
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            return f"Error getting article text from {url}: {e}"

def pprint_run_response(run_response: Union[RunResponse, Iterable[RunResponse]], markdown: bool = False, show_time: bool = False) -> None:
    from rich.box import ROUNDED
    from rich.json import JSON
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.status import Status
    from rich.table import Table
    from rich.console import Console
    console = Console()
    # If run_response is a single RunResponse, wrap it in a list to make it iterable
    if isinstance(run_response, RunResponse):
        single_response_content: Union[str, JSON, Markdown] = ""
        if isinstance(run_response.content, str):
            single_response_content = (Markdown(run_response.content) if markdown else run_response.get_content_as_string(indent=4))
        elif isinstance(run_response.content, BaseModel):
            try:
                single_response_content = JSON(run_response.content.model_dump_json(exclude_none=True), indent=2)
            except Exception as e:
                print(f"Failed to convert response to Markdown: {e}")
        else:
            try:
                single_response_content = JSON(json.dumps(run_response.content), indent=4)
            except Exception as e:
                print(f"Failed to convert response to string: {e}")
        table = Table(box=ROUNDED, border_style="blue", show_header=False)
        table.add_row(single_response_content)
        console.print(table)
    else:
        streaming_response_content: str = ""
        with Live(console=console) as live_log:
            status = Status("Working...", spinner="dots")
            live_log.update(status)
            response_timer = Timer()
            response_timer.start()
            for resp in run_response:
                if isinstance(resp, RunResponse) and isinstance(resp.content, str):
                    streaming_response_content += resp.content
                formatted_response = Markdown(streaming_response_content) if markdown else streaming_response_content
                table = Table(box=ROUNDED, border_style="blue", show_header=False)
                if show_time:
                    table.add_row(f"Response\n({response_timer.elapsed:.1f}s)", formatted_response)
                else:
                    table.add_row(formatted_response)
                live_log.update(table)
            response_timer.stop()
class StockAnalysis(BaseModel):
    symbol: str
    company_name: str
    analysis: str

stock_searcher = Agent(name="stock resercher",
    model=Ollama('llama3.1:8b'),
    response_model=StockAnalysis,
    role="在网上搜索股票信息。",
    tools=[
        YFinanceTools(stock_price=True,
            analyst_recommendations=True)
    ])

class CompanyAnalysis(BaseModel):
    company_name: str
    analysis: str

company_info_agent = Agent(name="company info researcher",
    model=Ollama('llama3.1:8b'),
    role="在网上搜索股票信息。",
    response_model=CompanyAnalysis,
    tools=[
        YFinanceTools(stock_price=False,
            company_info=True,
            company_news=True)
    ])

team = Team(name="股票研究团队",
    mode="route",
    model=Ollama('llama3.1:8b'),
    members=[stock_searcher, company_info_agent],
    markdown=True,

    debug_mode=True,
    show_members_responses=True)
response = team.run("NVDA目前的股价是多少？")
assert isinstance(response.content, StockAnalysis)
print(response.content)
response = team.run("关于NVDA的新闻是什么？")
assert isinstance(response.content, CompanyAnalysis)
print(response.content)

class Article(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")

class SearchResults(BaseModel):
    articles: list[Article]

class ScrapedArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")
    content: Optional[str] = Field(...,
        description="Content of the in markdown format if available. Return None if the content is not available or does not make sense.")

class ResearchReportGenerator(Workflow):

    description: str = dedent("""\
    Generate comprehensive research reports that combine academic rigor
    with engaging storytelling. This workflow orchestrates multiple AI agents to search, analyze,
    and synthesize information from diverse sources into well-structured reports.
    """)
    web_searcher: Agent = Agent(model=Ollama('llama3.1:8b'),
        tools=[DuckDuckGoTools()],
        description=dedent("""\
        You are ResearchBot-X, an expert at discovering and evaluating academic and scientific sources.\
        """),
        instructions=dedent("""\
        You're a meticulous research assistant with expertise in source evaluation! 🔍
        Search for 10-15 sources and identify the 5-7 most authoritative and relevant ones.
        Prioritize:
        - Peer-reviewed articles and academic publications
        - Recent developments from reputable institutions
        - Authoritative news sources and expert commentary
        - Diverse perspectives from recognized experts
        Avoid opinion pieces and non-authoritative sources.\
        """),
        response_model=SearchResults)
    article_scraper: Agent = Agent(model=Ollama('llama3.1:8b'),
        tools=[NewspaperTools()],
        description=dedent("""\
        You are ContentBot-X, an expert at extracting and structuring academic content.\
        """),
        instructions=dedent("""\
        You're a precise content curator with attention to academic detail! 📚
        When processing content:
           - Extract content from the article
           - Preserve academic citations and references
           - Maintain technical accuracy in terminology
           - Structure content logically with clear sections
           - Extract key findings and methodology details
           - Handle paywalled content gracefully
        Format everything in clean markdown for optimal readability.\
        """),
        response_model=ScrapedArticle)
    writer: Agent = Agent(model=Ollama('llama3.1:8b'),
        description=dedent("""\
        You are Professor X-2000, a distinguished AI research scientist combining academic rigor with engaging narrative style.\
        """),
        instructions=dedent("""\
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
        """),
        expected_output=dedent("""\
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
        """),
        markdown=True)

    def run(self,
        topic: str,
        use_search_cache: bool = True,
        use_scrape_cache: bool = True,
        use_cached_report: bool = True) -> Iterator[RunResponse]:
        """
        Generate a comprehensive news report on a given topic.
        This function orchestrates a workflow to search for articles, scrape their content,
        and generate a final report. It utilizes caching mechanisms to optimize performance.
        Args:
            topic (str): The topic for which to generate the news report.
            use_search_cache (bool, optional): Whether to use cached search results. Defaults to True.
            use_scrape_cache (bool, optional): Whether to use cached scraped articles. Defaults to True.
            use_cached_report (bool, optional): Whether to return a previously generated report on the same topic. Defaults to False.
        Returns:
            Iterator[RunResponse]: An stream of objects containing the generated report or status information.
        Steps:
        1. Check for a cached report if use_cached_report is True.
        2. Search the web for articles on the topic:
            - Use cached search results if available and use_search_cache is True.
            - Otherwise, perform a new web search.
        3. Scrape the content of each article:
            - Use cached scraped articles if available and use_scrape_cache is True.
            - Scrape new articles that aren't in the cache.
        4. Generate the final report using the scraped article contents.
        The function utilizes the `session_state` to store and retrieve cached data.
        """
        print(f"Generating a report on: {topic}")
        # Use the cached report if use_cached_report is True
        if use_cached_report:
            cached_report = self.get_cached_report(topic)
            if cached_report:
                yield RunResponse(content=cached_report, event=RunEvent.workflow_completed)
                return
        # Search the web for articles on the topic
        search_results: Optional[SearchResults] = self.get_search_results(topic, use_search_cache)
        # If no search_results are found for the topic, end the workflow
        if search_results is None or len(search_results.articles) == 0:
            yield RunResponse(event=RunEvent.workflow_completed,
                content=f"Sorry, could not find any articles on the topic: {topic}")
            return
        # Scrape the search results
        scraped_articles: Dict[str, ScrapedArticle] = self.scrape_articles(search_results, use_scrape_cache)
        # Write a research report
        yield from self.write_research_report(topic, scraped_articles)

    def get_cached_report(self, topic: str) -> Optional[str]:
        print("Checking if cached report exists")
        return self.session_state.get("reports", {}).get(topic)

    def add_report_to_cache(self, topic: str, report: str):
        print(f"Saving report for topic: {topic}")
        self.session_state.setdefault("reports", {})
        self.session_state["reports"][topic] = report
        # Save the report to the storage
        self.write_to_storage()

    def get_cached_search_results(self, topic: str) -> Optional[SearchResults]:
        print("Checking if cached search results exist")
        return self.session_state.get("search_results", {}).get(topic)

    def add_search_results_to_cache(self, topic: str, search_results: SearchResults):
        print(f"Saving search results for topic: {topic}")
        self.session_state.setdefault("search_results", {})
        self.session_state["search_results"][topic] = search_results.model_dump()
        # Save the search results to the storage
        self.write_to_storage()

    def get_cached_scraped_articles(self, topic: str) -> Optional[Dict[str, ScrapedArticle]]:
        print("Checking if cached scraped articles exist")
        return self.session_state.get("scraped_articles", {}).get(topic)

    def add_scraped_articles_to_cache(self, topic: str, scraped_articles: Dict[str, ScrapedArticle]):
        print(f"Saving scraped articles for topic: {topic}")
        self.session_state.setdefault("scraped_articles", {})
        self.session_state["scraped_articles"][topic] = scraped_articles
        # Save the scraped articles to the storage
        self.write_to_storage()

    def get_search_results(self, topic: str, use_search_cache: bool, num_attempts: int = 3) -> Optional[SearchResults]:
        # Get cached search_results from the session state if use_search_cache is True
        if use_search_cache:
            try:
                search_results_from_cache = self.get_cached_search_results(topic)
                if search_results_from_cache is not None:
                    search_results = SearchResults.model_validate(search_results_from_cache)
                    print(f"Found {len(search_results.articles)} articles in cache.")
                    return search_results
            except Exception as e:
                print(f"Could not read search results from cache: {e}")
        # If there are no cached search_results, use the web_searcher to find the latest articles
        for attempt in range(num_attempts):
            try:
                searcher_response: RunResponse = self.web_searcher.run(topic)
                if (searcher_response is not None
                    and searcher_response.content is not None
                    and isinstance(searcher_response.content, SearchResults)):
                    article_count = len(searcher_response.content.articles)
                    print(f"Found {article_count} articles on attempt {attempt + 1}")
                    # Cache the search results
                    self.add_search_results_to_cache(topic, searcher_response.content)
                    return searcher_response.content
                else:
                    print(f"Attempt {attempt + 1}/{num_attempts} failed: Invalid response type")
            except Exception as e:
                print(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")
        print(f"Failed to get search results after {num_attempts} attempts")
        return None

    def scrape_articles(self, search_results: SearchResults, use_scrape_cache: bool) -> Dict[str, ScrapedArticle]:
        scraped_articles: Dict[str, ScrapedArticle] = {}
        # Get cached scraped_articles from the session state if use_scrape_cache is True
        if use_scrape_cache:
            try:
                scraped_articles_from_cache = self.get_cached_scraped_articles(topic)
                if scraped_articles_from_cache is not None:
                    scraped_articles = scraped_articles_from_cache
                    print(f"Found {len(scraped_articles)} scraped articles in cache.")
                    return scraped_articles
            except Exception as e:
                print(f"Could not read scraped articles from cache: {e}")
        # Scrape the articles that are not in the cache
        for article in search_results.articles:
            if article.url in scraped_articles:
                print(f"Found scraped article in cache: {article.url}")
                continue
            article_scraper_response: RunResponse = self.article_scraper.run(article.url)
            if (article_scraper_response is not None
                and article_scraper_response.content is not None
                and isinstance(article_scraper_response.content, ScrapedArticle)):
                scraped_articles[article_scraper_response.content.url] = (article_scraper_response.content)
                print(f"Scraped article: {article_scraper_response.content.url}")
        # Save the scraped articles in the session state
        self.add_scraped_articles_to_cache(topic, scraped_articles)
        return scraped_articles

    def write_research_report(self, topic: str, scraped_articles: Dict[str, ScrapedArticle]) -> Iterator[RunResponse]:
        print("Writing research report")
        # Prepare the input for the writer
        writer_input = {
            "topic": topic,
            "articles": [v.model_dump() for v in scraped_articles.values()],
        }
        # Run the writer and yield the response
        yield from self.writer.run(json.dumps(writer_input, indent=4), stream=True)
        # Save the research report in the cache
        self.add_report_to_cache(topic, self.writer.run_response.content)

if __name__ == "__main__":
    from rich.prompt import Prompt
    example_topics = [
        "quantum computing breakthroughs 2024",
        "artificial consciousness research",
        "fusion energy developments",
        "space tourism environmental impact",
        "longevity research advances",
    ]
    topics_str = "\n".join(f"{i + 1}. {topic}" for i, topic in enumerate(example_topics))
    print(f"\n📚 Example Research Topics:\n{topics_str}\n")
    topic = "quantum computing breakthroughs 2024"
    url_safe_topic = topic.lower().replace(" ", "-")
    generate_research_report = ResearchReportGenerator(session_id=f"generate-report-on-{url_safe_topic}",
        storage=SqliteStorage(table_name="generate_research_report_workflow",
            db_file="workflows.db"))
    report_stream: Iterator[RunResponse] = generate_research_report.run(topic=topic,
        use_search_cache=True,
        use_scrape_cache=True,
        use_cached_report=True)
    pprint_run_response(report_stream, markdown=True)
