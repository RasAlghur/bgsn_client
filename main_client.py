# File 2
from flask import Flask, request, jsonify
from twocaptcha import TwoCaptcha
from celery import Celery
import requests
from httpx import AsyncClient, HTTPStatusError, TimeoutException, RequestError
import random
from collections import defaultdict
import pickle
from functools import lru_cache
import time
import aiohttp
import logging
import os
from os import getenv, path, makedirs
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
from aiohttp import ClientSession, TCPConnector, ClientTimeout
from aiohttp.client_exceptions import ClientError
import json
import re
from collections import deque
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, Timeout, RequestException
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from asyncio import TimeoutError, create_task, sleep, run, Semaphore, gather, new_event_loop, set_event_loop
from telegram import Update, ReplyKeyboardMarkup, BotCommand
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, KeyboardButton, ForceReply
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
import aiofiles
from multiprocessing import Pool, cpu_count
from openpyxl import Workbook
from itertools import cycle
from functools import wraps
import uuid
import threading

# In-memory job storage (replace with Redis for production)
jobs = {}

load_dotenv()
API_KEY = getenv('HELIUS_API_KEY')
BAGSCAN_API_KEY = getenv('BAGSCAN_API_KEY')
TELEGRAM_BOT_TOKEN = getenv('TELEGRAM_BOT_TOKEN')
app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = os.getenv('CELERY_BROKER_URL')
app.config['CELERY_RESULT_BACKEND'] = os.getenv('CELERY_RESULT_BACKEND')

# 2Captcha API key
CAPTCHA_API_KEY = getenv('TWOCAPTCHA_API_KEY')
solver = TwoCaptcha(CAPTCHA_API_KEY)

# Set your User-Agent
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

HOOK_SCRIPT = r"""
(() => {
  const interval = setInterval(() => {
    if (window.turnstile) {
      clearInterval(interval);
      window.turnstile.render = (elem, opts) => {
        window._cfTurnstileParams = {
          type: "TurnstileTaskProxyless",
          sitekey: opts.sitekey,
          url:     window.location.href,
          data:    opts.cData,
          pagedata:opts.chlPageData,
          action:  opts.action,
          userAgent: navigator.userAgent
        };
        window._cfTurnstileCallback = opts.callback;
        return "intercepted";
      };
    }
  }, 10);
})();
"""
ADDRESSES = [
    '4YduJ6ECHZmsas9iaEx4KtdzoSxcbrzSggYaP3P6hHjJ',
    '2cK5D5CQMQkB1A8Gn6UF4kLEgTpVDXqBSSdTYJZEHMZC',
    '7WVfgqr3ZEaoJCYkpEGdMR5tUh4czoyrw3Ff8pnmYkFu',
    '2sAyeuFhPzaRmeKKJzt2iQvwAWT7PKQ4hQVQGUZvardK',
    'BkmXCAh2Ca8x1NVuQuuYp9Gn1HRUkF9gMAdkTjNW8Spw',
    '2hmGETo6NUALnTcZBBPxRWProFPX7Z94YuzpZAKfdUMH'
]

ANALYSES_PERIOD = 30 #30 days
SOL_ADDR = "So11111111111111111111111111111111111111112"
USDT_ADDR = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"


SIGNATURES_URL = f"https://mainnet.helius-rpc.com/?api-key={API_KEY}"
TRANSACTIONS_URL = f"https://api.helius.xyz/v0/transactions?api-key={API_KEY}"

BASE_URL = f'https://api.helius.xyz/v0/addresses'
GET_ASSET_URL = f'https://mainnet.helius-rpc.com/?api-key={API_KEY}'
GET_NATIVE_BAL_URL = f'https://api.mainnet-beta.solana.com'

LINE_MULTIPLIER = 40
MAX_RETRIES = 5
SLEEP_TIME = 30
SHORT_SLEEP_TIME = 1
semaphore = Semaphore(10)

# State management: keep track of whether we're expecting a token address or holder addresses
USER_STATE = {}

web_Unlocker = {
    "server": "brd.superproxy.io:33335",
    "username": "brd-customer-hl_d768f7e0-zone-bag_unlocker",
    "password": "g9vjzx8i7zi9",
}

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

user_data = {}  # Initialize user data storage
greeted_users = {}
# Semaphore to limit concurrent usage of each proxy
PROXY_LIMIT = 5  # Adjust based on your proxy provider's limits
proxy_usage = {}

# Track proxy performance
proxy_stats = {}
proxy_cooldowns = {}
COOLDOWN_PERIOD = 30  # seconds
CACHE_TIMEOUT = 60 * 60  # 60 minutes in seconds

# In-memory cache for ultra-fast lookups
TOKEN_CACHE = {}

API_KEY = "b8f17a47139305b3a8ee6b34bc64615d1670f37df278f10dd04a3cd3440d94cb"  # Keep this secret!

def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config['CELERY_RESULT_BACKEND']
    )
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
         def __call__(self, *args, **kwargs):
             with app.app_context():
                 return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery

celery = make_celery(app)

# Function to verify the API key
def verify_api_key():
    client_api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    return client_api_key == API_KEY

async def take_error_screenshot(page, error_context, retry=None):
    """
    Save a screenshot with a filename based on the error context and current time.
    """
    screenshots_dir = "error_screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    retry_part = f"_retry{retry}" if retry is not None else ""
    screenshot_path = f"{screenshots_dir}/{error_context}_{timestamp}{retry_part}.png"
    try:
        await page.screenshot(path=screenshot_path, full_page=True)
        logger.info(f"Screenshot saved: {screenshot_path}")
    except Exception as ss_err:
        logger.error(f"Failed to take screenshot: {ss_err}")
    return screenshot_path

def ensure_directory_exists(directory):
    """Ensure that the specified directory exists."""
    if not path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# Function to check if a proxy is active (async)
async def is_proxy_active(proxy):
    url = "https://httpbin.org/ip"  # A simple endpoint to test the proxy
    try:
        ip, port, username, password = proxy.split(':')
        proxy_url = f"http://{username}:{password}@{ip}:{port}"

        async with ClientSession(timeout=ClientTimeout(total=3.0)) as session:
            async with session.get(url, proxy=proxy_url) as response:
                if response.status == 200:
                    return True
    except Exception as e:
        print(f"Error checking proxy {proxy}: {e}")
        return False
    return False

# Function to process a batch of proxies asynchronously
async def process_proxies_batch(proxies):
    tasks = [is_proxy_active(proxy) for proxy in proxies]
    results = await gather(*tasks)
    return results

# Function to run async tasks in multiprocessing
def run_async_batch(batch):
    loop = new_event_loop()
    set_event_loop(loop)
    return loop.run_until_complete(process_proxies_batch(batch))

# Function to process proxies in batches using multiprocessing
def process_proxies_in_batches(proxies, batch_size=100):
    batches = [proxies[i:i + batch_size] for i in range(0, len(proxies), batch_size)]
    active_proxies = []

    with Pool(cpu_count()) as pool:
        results = pool.map(run_async_batch, batches)

    for batch, result in zip(batches, results):
        for proxy, active in zip(batch, result):
            if active:
                active_proxies.append(proxy)

    return active_proxies

def divide_into_batches(tokens, batch_size):
    """
    Splits tokens into smaller batches of specified size.
    """
    for i in range(0, len(tokens), batch_size):
        yield tokens[i:i + batch_size]

# Fetch token metadata with retries
async def fetch_metadata(session, url, headers, payload, address, proxy_url):
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                async with session.post(url, headers=headers, json=payload, proxy=proxy_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and 'result' in data:
                            return address, data.get('result')
                        else:
                            print(f"Received empty or invalid data for address {address}")
                            return address, None
        except Exception as e:
            print(f"Attempt {attempt+1}/{MAX_RETRIES} failed for address {address}: {e}")
            await sleep(SLEEP_TIME)
    print(f"Failed to fetch metadata for address {address} after {MAX_RETRIES} attempts")
    return address, None

async def fetch_and_store_metadata(token_addresses, proxy=None):
    proxy_url = None
    if proxy:
        ip, port, username, password = proxy.split(':')
        proxy_url = f"http://{username}:{password}@{ip}:{port}"

    # Create the aiohttp session with or without proxy
    async with ClientSession() as session:
        tasks = []
        for address in token_addresses:
            url = GET_ASSET_URL
            headers = {'Content-Type': 'application/json'}
            payload = {
                'jsonrpc': '2.0',
                'id': 'my-id',
                'method': 'getAsset',
                'params': {
                    'id': address,
                    'displayOptions': {'showFungible': True}
                }
            }
            tasks.append(fetch_metadata(session, url, headers, payload, address, proxy_url))

        # Gather all the responses concurrently
        responses = await gather(*tasks)

        metadata_s = {}
        for address, metadata in responses:
            if metadata:  # Ensure metadata is not None
                metadata_s[address] = metadata
    return metadata_s

def parse_trade_string_broad(trade_str: str, mint_token: str):
    """
    Parses a trade string into its components using flexible pattern matching
    to handle various formats found on Solscan.
    
    Supports multiple formats including:
      1. Standard: <token_sold_amount><token_sold_symbol><token_bought_amount><mint_token>
      2. Inverted: <token_sold_amount><mint_token><token_bought_amount><token_bought_symbol>
      3. Complex formats with numbers and symbols in various positions
    
    Args:
      trade_str (str): The trade string to parse.
      mint_token (str): The token symbol for the minted token.
      
    Returns:
      dict: A dictionary with keys "token_sold_amount", "token_sold_symbol",
            "token_bought_amount", and "token_bought_symbol".
            
    Raises:
      ValueError: If the trade string cannot be parsed.
    """
    import re
    
    # Clean the input string and ensure it's properly formatted
    trade_str = trade_str.strip()
    
    # Helper: clean up number strings (remove commas)
    def clean_number(num_str: str):
        return num_str.replace(",", "")
    
    # Define regex patterns for various components
    # Pattern for numbers with optional commas and decimal points
    number_pattern = r"(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
    
    # Pattern for token symbols (letters, numbers, and some special characters)
    # Including numbers to handle tokens like "39a", "TIМ", etc.
    token_pattern = r"[A-Za-z0-9$._\-#@!]+"
    
    # Escape the mint token for safer regex
    mint_token_escaped = re.escape(mint_token)
    
    # Try to handle the format dynamically by analyzing the string
    try:
        # Case 1: Check if the string ends with mint_token
        if trade_str.endswith(mint_token):
            # Format: <sold_amount><sold_symbol><bought_amount><mint_token>
            pattern = rf"^({number_pattern})({token_pattern})({number_pattern})({mint_token_escaped})$"
            m = re.search(pattern, trade_str)
            if m:
                sold_amount_str, sold_token, bought_amount_str = m.group(1), m.group(2), m.group(3)
                
                return {
                    "token_sold_amount": clean_number(sold_amount_str),
                    "token_sold_symbol": sold_token,
                    "token_bought_amount": clean_number(bought_amount_str),
                    "token_bought_symbol": mint_token
                }
        
        # Case 2: Check if mint_token is in the middle of the string
        pattern = rf"({number_pattern})({mint_token_escaped})({number_pattern})({token_pattern})$"
        m = re.search(pattern, trade_str)
        if m:
            sold_amount_str, bought_amount_str, bought_token = m.group(1), m.group(3), m.group(4)
            
            return {
                "token_sold_amount": clean_number(sold_amount_str),
                "token_sold_symbol": mint_token,
                "token_bought_amount": clean_number(bought_amount_str),
                "token_bought_symbol": bought_token
            }
        
        # Case 3: General method - find all numbers and tokens
        # Split the string into sequences of digits and non-digits
        components = re.findall(rf"({number_pattern})|({token_pattern})", trade_str)
        
        # Filter out empty matches and flatten
        parts = [part for group in components for part in group if part]
        
        # Check if we have at least 4 parts (2 numbers, 2 tokens)
        if len(parts) >= 4:
            # Determine which parts are numbers and which are tokens
            numbers = []
            tokens = []
            
            for part in parts:
                # Check if the part is a number (starts with a digit)
                if re.match(r"^\d", part):
                    numbers.append(part)
                else:
                    tokens.append(part)
            
            # If we have 2 numbers and 2 tokens, we can parse
            if len(numbers) >= 2 and len(tokens) >= 2:
                # Identify which token is the mint token
                if mint_token in tokens:
                    mint_index = tokens.index(mint_token)
                    other_token_index = 1 - mint_index if len(tokens) == 2 else 0 if mint_index > 0 else 1
                    
                    # Determine the order based on positions
                    if parts.index(tokens[mint_index]) < parts.index(tokens[other_token_index]):
                        # Format: <sold_amount><mint_token><bought_amount><other_token>
                        return {
                            "token_sold_amount": clean_number(numbers[0]),
                            "token_sold_symbol": mint_token,
                            "token_bought_amount": clean_number(numbers[1]),
                            "token_bought_symbol": tokens[other_token_index]
                        }
                    else:
                        # Format: <sold_amount><other_token><bought_amount><mint_token>
                        return {
                            "token_sold_amount": clean_number(numbers[0]),
                            "token_sold_symbol": tokens[other_token_index],
                            "token_bought_amount": clean_number(numbers[1]),
                            "token_bought_symbol": mint_token
                        }
                else:
                    # If mint token isn't found in tokens, assume it's the second token position
                    return {
                        "token_sold_amount": clean_number(numbers[0]),
                        "token_sold_symbol": tokens[0],
                        "token_bought_amount": clean_number(numbers[1]),
                        "token_bought_symbol": tokens[1] if len(tokens) > 1 else mint_token
                    }
                    
        # Case 4: Special case for more complex patterns
        # Handle formats like "198,3005.57,667.080378SAG3" where "5.5" is the mint token
        if mint_token in trade_str:
            # Split at the mint token
            parts = trade_str.split(mint_token)
            if len(parts) == 2:
                # Get the amounts and symbols
                # Extract first number from first part
                first_num_match = re.search(rf"^{number_pattern}", parts[0])
                if first_num_match:
                    first_amount = first_num_match.group(0)
                    
                    # Extract number and token from second part
                    second_match = re.search(rf"({number_pattern})({token_pattern})$", parts[1])
                    if second_match:
                        second_amount, second_token = second_match.group(1), second_match.group(2)
                        
                        return {
                            "token_sold_amount": clean_number(first_amount),
                            "token_sold_symbol": mint_token,
                            "token_bought_amount": clean_number(second_amount),
                            "token_bought_symbol": second_token
                        }
        
        # Last resort: Try to extract structured data from the unstructured string
        nums = re.findall(number_pattern, trade_str)
        if len(nums) >= 2:
            # Extract token symbols by removing numbers, common separators, and whitespace
            remaining = re.sub(rf"{number_pattern}|[,.\s]", " ", trade_str).strip()
            tokens_found = re.findall(r"[A-Za-z0-9$]+", remaining)
            
            if len(tokens_found) >= 1:
                # If mint_token matches one of the found tokens
                if mint_token in tokens_found:
                    other_tokens = [t for t in tokens_found if t != mint_token]
                    other_token = other_tokens[0] if other_tokens else "UNKNOWN"
                    
                    # Determine order based on position in the original string
                    if trade_str.find(mint_token) < trade_str.find(other_token):
                        return {
                            "token_sold_amount": clean_number(nums[0]),
                            "token_sold_symbol": mint_token,
                            "token_bought_amount": clean_number(nums[1]),
                            "token_bought_symbol": other_token
                        }
                    else:
                        return {
                            "token_sold_amount": clean_number(nums[0]),
                            "token_sold_symbol": other_token,
                            "token_bought_amount": clean_number(nums[1]),
                            "token_bought_symbol": mint_token
                        }
                else:
                    # If mint_token not found, make best guess based on position
                    if len(tokens_found) >= 2:
                        return {
                            "token_sold_amount": clean_number(nums[0]),
                            "token_sold_symbol": tokens_found[0],
                            "token_bought_amount": clean_number(nums[1]),
                            "token_bought_symbol": tokens_found[1]
                        }
                    else:
                        # Assume the only token found is the sold token
                        return {
                            "token_sold_amount": clean_number(nums[0]),
                            "token_sold_symbol": tokens_found[0],
                            "token_bought_amount": clean_number(nums[1]),
                            "token_bought_symbol": mint_token
                        }
        
        # If we reach here, we couldn't parse the trade string
        raise ValueError(f"Could not parse trade string: '{trade_str}' with mint token '{mint_token}'")
        
    except Exception as e:
        # Provide detailed error for debugging
        raise ValueError(f"Error parsing trade string '{trade_str}': {str(e)}")

async def save_accumulated_data(mint_address, all_data):
    """Save all accumulated data to a single file after each page"""
    os.makedirs("results", exist_ok=True)
    output_path = f"results/{mint_address}_data.json"
    
    # Use a temporary file to prevent data loss if the save operation is interrupted
    temp_path = f"{output_path}.temp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4)
    
    # Rename the temp file to the final file name
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_path, output_path)
    
    logger.info(f"Updated data saved to {output_path} ({len(all_data)} total records)")

async def load_existing_data(mint_address):
    """Load any existing data to continue where we left off"""
    output_path = f"results/{mint_address}_data.json"
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                logger.info(f"Loaded {len(existing_data)} existing records from {output_path}")
                return existing_data
        except json.JSONDecodeError:
            logger.warning(f"Could not parse existing data file {output_path}, starting fresh")
    return []

async def process_row(row, row_index, mint_address, check_symbol):
    """Process a single table row and extract data concurrently"""
    if row_index == 0:  # Skip header row
        return None
    
    cells = row.locator("td")
    cell_count = await cells.count()

     # Helper: clean up number strings (remove commas)
    def clean_number(num_str: str):
        return num_str.replace(",", "")
    
    
    # Create tasks to extract all cell text content in parallel
    cell_text_tasks = []
    for j in range(cell_count):
        cell = cells.nth(j)
        cell_text_tasks.append(cell.text_content())
    
    # Await all cell text tasks simultaneously
    cell_texts_raw = await gather(*cell_text_tasks)
    cell_texts = [text.strip() for text in cell_texts_raw]
    
    # Default values in case extraction fails
    sell_amount = sell_symbol = buy_amount = buy_symbol = "N/A"
    
    # Parse amounts and symbols from cell index 5
    if len(cell_texts) > 5:
        try:
            amount_cell = cells.nth(5)
            container = amount_cell.locator("div.flex.flex-col.gap-1.items-stretch.justify-start")
            row_inner = container.locator("div.flex.gap-1.flex-row.items-center.justify-start.flex-nowrap")
            
            # Create tasks for parallel extraction of all needed elements
            extraction_tasks = [
                amount_cell.text_content(),  # full text content
                row_inner.locator("div").nth(0).text_content(),  # sell amount
                row_inner.locator("span.whitespace-nowrap").nth(0).locator("a").text_content(),  # sell symbol
                row_inner.locator("span.whitespace-nowrap").nth(1).locator("a").text_content()  # buy symbol
            ]
            
            # Execute all extraction tasks in parallel
            results = await gather(*extraction_tasks)
            
            # Parse the results
            full_text = results[0].strip()
            sell_amount = results[1].strip()
            sell_symbol = results[2].strip()
            buy_symbol = results[3].strip()
            
            # Extract buy amount by removing known parts
            temp = full_text.replace(sell_amount, "", 1)
            temp = temp.replace(sell_symbol, "", 1)
            temp = temp.replace(buy_symbol, "", 1)
            buy_amount = temp.strip()

            
            
            logger.info(f"Sell Amount: {sell_amount}, Sell Symbol: {sell_symbol}, Buy Amount: {buy_amount}, Buy Symbol: {buy_symbol}")
        except Exception as e:
            logger.warning(f"Failed to parse buy/sell info on row {row_index}: {e}")
    
    signature = cell_texts[1] if len(cell_texts) > 1 else "N/A"
    trade_time = cell_texts[2] if len(cell_texts) > 2 else "N/A"
    transaction_by = cell_texts[4] if len(cell_texts) > 4 else "N/A"
    txn_usd_value_s = cell_texts[6] if len(cell_texts) > 6 else "N/A"
    token_sold_amount = clean_number(sell_amount)
    token_sold = sell_symbol
    token_bought_amount = clean_number(buy_amount)
    token_bought = buy_symbol
    txn_usd_value = clean_number(txn_usd_value_s)
    
    # Clean extracted data
    signature = signature.strip()
    trade_time = trade_time.strip()
    transaction_by = transaction_by.strip()
    txn_usd_value = txn_usd_value.strip()

    # Construct the row data dictionary
    txn_dict = {
        "mint_addr": mint_address,
        "signature": signature,
        "trade_time": trade_time,
        "transaction_by": transaction_by,
        "token_sold": token_sold,
        "token_sold_amount": token_sold_amount,
        "token_bought": token_bought,
        "token_bought_amount": token_bought_amount,
        "txn_usd_value": txn_usd_value,
        "mint_symbol": check_symbol,
    }
    
    return txn_dict

async def process_rows_in_batches(rows, mint_address, check_symbol, batch_size=100):
    """Process table rows in parallel batches for faster extraction with timing metrics"""
    row_count = len(rows)
    all_transactions = []
    total_batch_time = 0
    
    logger.info(f"Starting batch processing of {row_count} rows with batch size {batch_size}")
    
    # Process rows in batches
    for start_idx in range(0, row_count, batch_size):
        batch_start_time = time.time()
        
        end_idx = min(start_idx + batch_size, row_count)
        batch_tasks = []
        
        for row_idx in range(start_idx, end_idx):
            batch_tasks.append(process_row(rows[row_idx], row_idx, mint_address, check_symbol))
        
        batch_results = await gather(*batch_tasks)
        valid_results = [result for result in batch_results if result]
        all_transactions.extend(valid_results)
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        total_batch_time += batch_duration
        
        # Log batch timing information
        batch_size_actual = end_idx - start_idx
        rows_per_second = batch_size_actual / batch_duration if batch_duration > 0 else 0
        logger.info(f"Batch {start_idx // batch_size + 1}: Processed {len(valid_results)}/{batch_size_actual} rows in {batch_duration:.2f}s ({rows_per_second:.2f} rows/s)")
    
    # Log overall batch processing metrics
    avg_time_per_row = total_batch_time / row_count if row_count > 0 else 0
    logger.info(f"Completed processing {row_count} rows in {total_batch_time:.2f}s (avg {avg_time_per_row:.4f}s per row)")
    
    return all_transactions

proxy_config = {
    "server": "unblock.oxylabs.io:60000",
    "username": "devras_fcwXF",
    "password": "Keepmyoxysafe1+"
}

async def fetch_transactions_batch(session, batch, url, headers, proxy):
    """Fetch transactions for a single batch asynchronously."""
    body = {"transactions": batch}
    try:
        async with session.post(url, json=body, headers=headers, proxy=proxy, timeout=10) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Failed to fetch transactions. Status: {response.status}, Response: {await response.text()}")
                return []
    except Exception as e:
        print(f"Error fetching transactions: {e}")
        return []

# START OF THE TOKEN REPORTING FUNCTIONALITY
def get_cache_filename(token):
    """Build a cache filename for a given token within cache directory."""
    # Use the first 2 characters of token to create subdirectories
    # This reduces the number of files in a single directory
    prefix = token[:2] if len(token) >= 2 else token
    directory = os.path.join("cache_tokens", prefix)
    ensure_directory_exists(directory)
    return os.path.join(directory, f"cache_{token}.pkl")  # Using pickle format

def get_cache_index_filename():
    """Get the filename for the cache index."""
    directory = "cache_tokens"
    ensure_directory_exists(directory)
    return os.path.join(directory, "cache_index.pkl")

@lru_cache(maxsize=1000)
def cache_file_exists(token):
    """Check if cache file exists with LRU caching for better performance."""
    filename = get_cache_filename(token)
    return os.path.exists(filename)

def update_cache_index(token, expiry_time):
    """Update the cache index with a token and its expiry time."""
    index_file = get_cache_index_filename()
    try:
        if os.path.exists(index_file):
            with open(index_file, "rb") as f:
                cache_index = pickle.load(f)
        else:
            cache_index = {}
        
        cache_index[token] = expiry_time
        
        # Clean up expired entries
        current_time = time.time()
        cache_index = {k: v for k, v in cache_index.items() if v > current_time}
        
        with open(index_file, "wb") as f:
            pickle.dump(cache_index, f)
    except Exception as e:
        logger.error(f"Error updating cache index: {e}")

def get_cached_token_data(token):
    """
    Get cached token data with optimized in-memory lookup first.
    Returns the cached data if valid, or None otherwise.
    """
    # Check in-memory cache first (fastest)
    if token in TOKEN_CACHE:
        cache_entry = TOKEN_CACHE[token]
        if time.time() < cache_entry["expiry"]:
            logger.info(f"Using in-memory cached data for token {token}")
            return cache_entry["data"]
        else:
            # Remove expired entry
            del TOKEN_CACHE[token]
    
    # If not in memory, check file cache
    if not cache_file_exists(token):
        return None
        
    filename = get_cache_filename(token)
    mod_time = os.path.getmtime(filename)
    
    if time.time() - mod_time < CACHE_TIMEOUT:
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                # Update in-memory cache for future use
                TOKEN_CACHE[token] = {
                    "data": data,
                    "expiry": mod_time + CACHE_TIMEOUT
                }
                logger.info(f"Using file cached data for token {token}")
                return data
        except Exception as e:
            logger.error(f"Error reading cache file for token {token}: {e}")
    
    return None

def save_cached_token_data(token, data):
    """
    Saves the token data to both in-memory and file cache for faster access.
    """
    # Calculate expiry time
    expiry_time = time.time() + CACHE_TIMEOUT
    
    # Update in-memory cache
    TOKEN_CACHE[token] = {
        "data": data,
        "expiry": expiry_time
    }
    
    # Also save to file for persistence
    filename = get_cache_filename(token)
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        
        # Update the cache index
        update_cache_index(token, expiry_time)
        logger.info(f"Cached data saved for token {token}")
    except Exception as e:
        logger.error(f"Error saving cache file for token {token}: {e}")

async def check_cache_exists_batch(tokens):
    """
    Check if cache files exist for multiple tokens simultaneously.
    Returns a dictionary of {token: cached_data} for tokens with valid cache.
    """
    # First check in-memory cache
    results = {}
    tokens_to_check = []
    
    for token in tokens:
        if token in TOKEN_CACHE and time.time() < TOKEN_CACHE[token]["expiry"]:
            results[token] = TOKEN_CACHE[token]["data"]
        else:
            tokens_to_check.append(token)
    
    # For remaining tokens, check file system in parallel
    if tokens_to_check:
        tasks = [create_task(check_file_cache(token)) for token in tokens_to_check]
        cache_results = await gather(*tasks)
        
        for token, data in zip(tokens_to_check, cache_results):
            if data:
                results[token] = data
    
    return results

async def check_file_cache(token):
    """Check file cache for a single token asynchronously"""
    if not cache_file_exists(token):
        return None
        
    filename = get_cache_filename(token)
    mod_time = os.path.getmtime(filename)
    
    if time.time() - mod_time < CACHE_TIMEOUT:
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                # Update in-memory cache
                TOKEN_CACHE[token] = {
                    "data": data,
                    "expiry": mod_time + CACHE_TIMEOUT
                }
                return data
        except Exception as e:
            logger.error(f"Error reading cache file for token {token}: {e}")
    
    return None


async def fetch_token_data(token_m, proxy, semaphore, results, failed_tokens, retry_attempt=0):
    """
    Fetch token data from the RugCheck report API with improved error handling
    and proxy management. On a successful fetch, the cache is updated.
    """
    global proxy_stats, proxy_cooldowns

    # Check if proxy is on cooldown
    current_time = time.time()
    if proxy in proxy_cooldowns and proxy_cooldowns[proxy] > current_time:
        logger.info(f"Proxy {proxy} is on cooldown, skipping")
        failed_tokens.append((token_m, retry_attempt))
        return

    url = f"https://api.rugcheck.xyz/v1/tokens/{token_m}/report"

    async with semaphore:
        try:
            ip, port, username, password = proxy.split(':')
            proxy_url = f"http://{username}:{password}@{ip}:{port}"
            logger.info(f"Using proxy {proxy_url} for token {token_m} (attempt {retry_attempt+1})")

            connector = TCPConnector(keepalive_timeout=60)
            timeout = ClientTimeout(total=10.0, connect=5.0)

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "Cache-Control": "no-cache"
            }

            async with ClientSession(timeout=timeout, connector=connector) as session:
                start_time = time.time()
                async with session.get(url, proxy=proxy_url, headers=headers, ssl=False) as response:
                    response_time = time.time() - start_time

                    if proxy not in proxy_stats:
                        proxy_stats[proxy] = {"success": 0, "fail": 0, "avg_time": 0}

                    if response.status == 200:
                        proxy_stats[proxy]["success"] += 1
                        proxy_stats[proxy]["avg_time"] = (proxy_stats[proxy]["avg_time"] * (proxy_stats[proxy]["success"] - 1) + response_time) / proxy_stats[proxy]["success"]

                        data = await response.json()
                        score_norm = data.get("score_normalised", "N/A")
                        markets = data.get("markets", [])
                        basePrice = "N/A"
                        quotePrice = "N/A"
                        if markets and isinstance(markets, list) and len(markets) > 0:
                            lp = markets[0].get("lp", {})
                            basePrice = lp.get("basePrice", "N/A")
                            quotePrice = lp.get("quotePrice", "N/A")

                        token_info = data.get("token", {})
                        if token_info and isinstance(token_info, dict):
                            decimals = token_info.get("decimals", "N/A")
                            supply = token_info.get("supply", "N/A")
                        else:
                            decimals = "N/A"
                            supply = "N/A"

                        token_result = {
                            "score_normalised": score_norm,
                            "basePrice": basePrice,
                            "quotePrice": quotePrice,
                            "supply": supply,
                            "decimals": decimals,
                            "proxy_used": proxy_url,
                            "response_time": round(response_time, 2)
                        }
                        results[token_m] = token_result

                        # Save fetched data to cache
                        save_cached_token_data(token_m, token_result)
                        logger.info(f"Successfully fetched data for {token_m} in {round(response_time, 2)}s")
                    elif response.status == 429:
                        logger.warning(f"Rate limit hit for token {token_m} with proxy {proxy}. Adding to retry queue.")
                        proxy_stats[proxy]["fail"] += 1
                        proxy_cooldowns[proxy] = current_time + COOLDOWN_PERIOD
                        failed_tokens.append((token_m, retry_attempt))
                    else:
                        logger.error(f"Error fetching data for token {token_m}: Status {response.status}")
                        proxy_stats[proxy]["fail"] += 1
                        failed_tokens.append((token_m, retry_attempt))
        except Exception as e:
            if proxy in proxy_stats:
                proxy_stats[proxy]["fail"] += 1
            else:
                proxy_stats[proxy] = {"success": 0, "fail": 1, "avg_time": 0}

            logger.error(f"Error fetching data for token {token_m} with proxy {proxy}: {str(e)}")
            failed_tokens.append((token_m, retry_attempt))
            if retry_attempt >= MAX_RETRIES:
                results[token_m] = {
                    "score_normalised": "N/A",
                    "basePrice": "N/A",
                    "quotePrice": "N/A",
                    "supply": "N/A",
                    "decimals": "N/A",
                    "proxy_used": proxy,
                    "error": str(e)
                }

async def check_tokens_for_report(token_mint, active_proxies):
    """
    Process tokens with smarter retry logic and proxy selection.
    Uses optimized cache checking for faster performance.
    """
    results = {}
    failed_tokens = []
    semaphore = Semaphore(5)  # Limit concurrent requests
    tasks = []
    
    # Check all tokens against cache in one batch operation
    cached_results = await check_cache_exists_batch(token_mint)
    results.update(cached_results)
    
    # Only process tokens that weren't found in cache
    tokens_to_fetch = [token for token in token_mint if token not in cached_results]
    
    if not tokens_to_fetch:
        logger.info("All tokens found in cache, no API requests needed")
        return results
        
    logger.info(f"Found {len(cached_results)} tokens in cache, fetching {len(tokens_to_fetch)} from API")
    
    for token in tokens_to_fetch:
        proxy = select_best_proxy(active_proxies)
        tasks.append(fetch_token_data(token, proxy, semaphore, results, failed_tokens, 0))
    
    if tasks:
        await gather(*tasks)

    # Process any tokens that failed to fetch (with retries)
    retry_attempt = 0
    while failed_tokens and retry_attempt < MAX_RETRIES:
        retry_attempt += 1
        logger.info(f"Starting retry attempt {retry_attempt} for {len(failed_tokens)} tokens")
        await sleep(2 ** retry_attempt)

        retry_batch = failed_tokens.copy()
        failed_tokens = []
        new_tasks = []

        for token, prev_attempts in retry_batch:
            # Check cache again before retrying
            cached_data = get_cached_token_data(token)
            if cached_data:
                results[token] = cached_data
                continue
            proxy = select_different_proxy(active_proxies, results.get(token, {}).get("proxy_used", ""))
            new_tasks.append(fetch_token_data(token, proxy, semaphore, results, failed_tokens, prev_attempts + 1))

        if new_tasks:
            await gather(*new_tasks)

    for token, _ in failed_tokens:
        if token not in results:
            results[token] = {
                "score_normalised": "N/A",
                "basePrice": "N/A",
                "quotePrice": "N/A",
                "supply": "N/A",
                "decimals": "N/A",
                "proxy_used": "max_retries_reached",
                "error": "Maximum retries reached"
            }

    logger.info(f"Proxy performance stats: {json.dumps(proxy_stats, indent=2)}")
    return results

def select_best_proxy(proxies):
    """
    Select the best performing proxy based on success rate and response time.
    """
    if not proxy_stats:
        return random.choice(proxies)
    current_time = time.time()
    available_proxies = [p for p in proxies if p not in proxy_cooldowns or proxy_cooldowns[p] <= current_time]
    if not available_proxies:
        return min(proxies, key=lambda p: proxy_cooldowns.get(p, current_time))
    ranked_proxies = []
    for proxy in available_proxies:
        if proxy in proxy_stats:
            stats = proxy_stats[proxy]
            total = stats["success"] + stats["fail"]
            success_rate = stats["success"] / total if total > 0 else 0
            score = success_rate * (1 / (stats["avg_time"] + 0.1))
            ranked_proxies.append((proxy, score))
        else:
            ranked_proxies.append((proxy, 0.5))
    if ranked_proxies:
        total_score = sum(score for _, score in ranked_proxies)
        if total_score > 0:
            r = random.random() * total_score
            cumulative = 0
            for proxy, score in ranked_proxies:
                cumulative += score
                if cumulative >= r:
                    return proxy
    return random.choice(available_proxies)

def select_different_proxy(proxies, previous_proxy):
    """
    Select a different proxy than the one previously used.
    """
    if len(proxies) == 1:
        return proxies[0]
    available = [p for p in proxies if p != previous_proxy]
    if not available:
        return random.choice(proxies)
    return select_best_proxy(available)

def run_async_batch_report(batch, active_proxies):
    loop = new_event_loop()
    set_event_loop(loop)
    return loop.run_until_complete(check_tokens_for_report(batch, active_proxies))

async def process_tokens_in_batches(token_mint, active_proxies, batch_size=10):
    """
    Process tokens in optimized batches with faster cache checking.
    Increased batch size from 3 to 10 for better performance.
    """
    # First check all tokens against cache in one operation
    cached_results = await check_cache_exists_batch(token_mint)
    
    # Only process tokens that weren't found in cache
    tokens_to_fetch = [token for token in token_mint if token not in cached_results]
    
    logger.info(f"Found {len(cached_results)} tokens in cache out of {len(token_mint)} total tokens")
    
    if not tokens_to_fetch:
        logger.info("All tokens found in cache, no API requests needed")
        return cached_results
        
    # Process remaining tokens in batches
    def divide_into_batches(tokens, batch_size):
        for i in range(0, len(tokens), batch_size):
            yield tokens[i:i+batch_size]
    
    token_batches = list(divide_into_batches(tokens_to_fetch, batch_size))
    batch_results = {}
    
    for batch in token_batches:
        results = await check_tokens_for_report(batch, active_proxies)
        batch_results.update(results)
        await sleep(0.5)  # shorter pause between batches
    
    # Combine cached and fresh results
    combined_results = {**cached_results, **batch_results}
    return combined_results

# END OF TOKEN REPORTS

def is_within_n_days(time_str, analyses_period=7):
    """
    Determine if a UTC time string represents a time within the last analyses_period days.
    
    Args:
        time_str (str): UTC datetime string in format "MM-DD-YYYY HH:MM:SS"
        analyses_period (int): Maximum threshold in days.
        
    Returns:
        bool: True if the transaction is within the last analyses_period days, otherwise False.
    """
    try:
        # Parse the UTC datetime string
        transaction_time = datetime.strptime(time_str, '%m-%d-%Y %H:%M:%S')
        current_time = datetime.utcnow()  # Current time in UTC
        time_diff = current_time - transaction_time
        return time_diff.days <= analyses_period
    except ValueError as e:
        logger.error(f"Error parsing datetime: {e}")
        return True  # Default to considering it recent if parsing fails

def convert_utc_to_formats(utc_time_str):
    """
    Convert a UTC time string to multiple formats: relative time and unix timestamp.
    
    Args:
        utc_time_str (str): UTC datetime string in format "MM-DD-YYYY HH:MM:SS"
        
    Returns:
        dict: Dictionary containing original UTC string, relative time, and unix timestamp
    """
    try:
        # Parse the UTC datetime string
        transaction_time = datetime.strptime(utc_time_str, '%m-%d-%Y %H:%M:%S')
        current_time = datetime.utcnow()  # Current time in UTC
        
        # Calculate time difference
        time_diff = current_time - transaction_time
        
        # Convert to Unix timestamp (seconds since epoch)
        unix_timestamp = int(transaction_time.timestamp())
        
        # Generate relative time string
        if time_diff.days > 365:
            years = time_diff.days // 365
            relative_time = f"{years} {'year' if years == 1 else 'years'} ago"
        elif time_diff.days > 30:
            months = time_diff.days // 30
            relative_time = f"{months} {'month' if months == 1 else 'months'} ago"
        elif time_diff.days > 0:
            relative_time = f"{time_diff.days} {'day' if time_diff.days == 1 else 'days'} ago"
        elif time_diff.seconds // 3600 > 0:
            hours = time_diff.seconds // 3600
            relative_time = f"{hours} {'hour' if hours == 1 else 'hours'} ago"
        elif time_diff.seconds // 60 > 0:
            minutes = time_diff.seconds // 60
            relative_time = f"{minutes} {'minute' if minutes == 1 else 'minutes'} ago"
        else:
            relative_time = f"{time_diff.seconds} {'second' if time_diff.seconds == 1 else 'seconds'} ago"
        
        return {
            "utc_time": utc_time_str,
            "relative_time": relative_time,
            "unix_timestamp": unix_timestamp
        }
    except ValueError as e:
        logger.error(f"Error converting datetime: {e}")
        return {
            "utc_time": utc_time_str,
            "relative_time": "unknown",
            "unix_timestamp": None
        }

async def check_and_solve_cloudflare(page, context) -> bool:
    """Navigate (or reload) the page, solve Turnstile if present, and ensure bypass."""
    logger.info("Injecting Turnstile hook & loading page...")
    await context.add_init_script(HOOK_SCRIPT)

    # Load or reload the page
    try:
        await page.reload(wait_until="domcontentloaded")
    except:
        await page.goto(page.url, wait_until="domcontentloaded")

    # Give the hook a moment
    await sleep(1)

    # If no widget params found, assume no Turnstile
    params = await page.evaluate("window._cfTurnstileParams || null")
    if not params:
        logger.info("No Turnstile detected.")
        return True

    logger.info(f"Captured Turnstile params: {json.dumps(params, indent=2)}")

    # Solve via 2Captcha
    try:
        result = solver.turnstile(
            type=params["type"],
            sitekey=params["sitekey"],
            url=params["url"],
            data=params["data"],
            pagedata=params["pagedata"],
            action=params["action"],
            useragent=params["userAgent"]
        )
        token = result["code"]
        logger.info("2Captcha returned token.")
    except Exception as e:
        logger.error(f"2Captcha solve failed: {e}")
        await take_error_screenshot(page, "captcha_solve_failed")
        return False

    # Inject token
    injected = await page.evaluate(
        """(t) => {
             if (window._cfTurnstileCallback) {
               window._cfTurnstileCallback(t);
               return true;
             }
             return false;
           }""",
        token
    )
    if injected is not True:
        logger.error("Token injection failed.")
        await take_error_screenshot(page, "token_injection_failed")
        return False

    logger.info("Token injected; reloading to finalize bypass.")
    try:
        await page.wait_for_timeout(1000)
        await page.reload(wait_until="networkidle")
    except Exception:
        await page.goto(page.url, wait_until="networkidle")

    # Verify bypass
    content = await page.content()
    if "cf-browser-verification" in content.lower() or "turnstile" in content.lower():
        logger.error("Still blocked after injection.")
        await take_error_screenshot(page, "post_injection_blocked")
        return False

    logger.info("Successfully bypassed Cloudflare Turnstile!")
    return True

# -------------------------------------------------------------------
# Scraping functions
# -------------------------------------------------------------------
async def get_all_signatures_01(mint_address, active_proxies=None, proxy=None):
    URL = "https://solscan.io"
    pages_counter = 0
    total_processing_time = 0

    metadata = await fetch_and_store_metadata([mint_address], proxy)
    symbol = metadata.get(mint_address, {}).get("content", {}).get("metadata", {}).get("symbol", "N/A")

    all_data = await load_existing_data(mint_address)
    existing_signatures = {item["signature"] for item in all_data if item.get("signature") and item["signature"] != "N/A"}

    new_data = []
    found_existing = False

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        context = await browser.new_context(ignore_https_errors=True, user_agent=USER_AGENT)
        page = await context.new_page()
        target_url = f"{URL}/token/{mint_address}?activity_type=ACTIVITY_TOKEN_SWAP&activity_type=ACTIVITY_AGG_TOKEN_SWAP#defiactivities"
        await page.goto(target_url, timeout=90000)

        if not await check_and_solve_cloudflare(page, context):
            logger.error("Failed to bypass Cloudflare. Aborting.")
            return all_data

        await page.wait_for_selector(".caption-bottom", timeout=60000)
        total_elem = await page.get_by_text("activities(s)").first.text_content()
        total = int(re.search(r"Total\s([\d,]+)", total_elem).group(1).replace(",", ""))
        rows_per_page = 100
        total_pages = (total + rows_per_page - 1) // rows_per_page
        logger.info(f"Total activities: {total} | Pages: {total_pages}")

        # Set 100 rows/page
        await page.locator("button[role='combobox']").click()
        await page.get_by_label("100", exact=True).get_by_text("100").click()
        await page.wait_for_timeout(3000)

        while pages_counter < total_pages:
            start = time.time()
            rows = await page.evaluate("""
                () => Array.from(document.querySelectorAll(".caption-bottom tr"))
                      .map(r => Array.from(r.querySelectorAll("td")).map(td => td.innerText.trim()))
            """)
            processed = []
            for row in rows:
                if len(row) < 7: continue
                sig = row[1].strip()
                if sig in existing_signatures: continue
                data = {
                    "mint_addr": mint_address,
                    "mint_symbol": symbol,
                    "signature": sig,
                    "trade_time": row[2].strip(),
                    "transaction_by": row[4].strip(),
                    **dict(zip(
                        ["token_sold_amount","token_sold","token_bought_amount","token_bought"],
                        row[5].splitlines()[:4]
                    )),
                    "txn_usd_value": row[6].strip()
                }
                processed.append(data)
                existing_signatures.add(sig)

            if processed:
                new_data.extend(processed)
                await save_accumulated_data(mint_address, new_data + all_data)
            else:
                found_existing = True
                break

            elapsed = time.time() - start
            logger.info(f"Page {pages_counter+1} done in {elapsed:.2f}s")

            if pages_counter < total_pages - 1:
                await page.locator("div:nth-child(2) > button:nth-child(4)").click()
                pages_counter += 1
                await page.wait_for_timeout(5000)
                if pages_counter % 5 == 0:
                    await check_and_solve_cloudflare(page, context)
            else:
                break

        await save_accumulated_data(mint_address, new_data + all_data)
        return new_data + all_data

async def get_all_account_signatures_01(address, update, proxy=None, analyses_period=7, active_proxies=None):
    URL = "https://solscan.io"
    pages_counter = 0
    total_processing_time = 0
    new_data = []
    found_old = False

    cutoff = datetime.utcnow() - timedelta(days=analyses_period)

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        context = await browser.new_context(ignore_https_errors=True, user_agent=USER_AGENT)
        page = await context.new_page()
        target = f"{URL}/account/{address}?activity_type=ACTIVITY_TOKEN_SWAP&activity_type=ACTIVITY_AGG_TOKEN_SWAP#defiactivities"
        await page.goto(target)

        if not await check_and_solve_cloudflare(page, context):
            logger.error("Failed Cloudflare bypass. Aborting.")
            return []

        await page.wait_for_selector(".caption-bottom", timeout=60000)
        total_elem = await page.get_by_text("activities(s)").first.text_content()
        total = int(re.search(r"Total\s([\d,]+)", total_elem).group(1).replace(",", ""))
        total_pages = (total + 99) // 100
        logger.info(f"Total activities: {total} | Pages: {total_pages}")

        # Set rows per page
        await page.locator("button[role='combobox']").click()
        await page.get_by_label("100", exact=True).get_by_text("100").click()
        await page.wait_for_timeout(10000)

        # Sort by time (desc)
        await page.locator("th:has-text('Time') .flex-row").first.click()
        await page.wait_for_timeout(2000)

        while pages_counter < total_pages:
            start = time.time()
            rows = await page.evaluate("""
                () => Array.from(document.querySelectorAll(".caption-bottom tr"))
                      .map(r => Array.from(r.querySelectorAll("td")).map(td => {
                        return {
                          text: td.innerText.trim(),
                          hrefs: Array.from(td.querySelectorAll("a")).map(a=>a.href)
                        };
                      }))
            """)
            for row in rows:
                if len(row) < 7: continue

                # parse time and skip old
                rel = row[2]["text"]
                if not is_within_n_days(rel, analyses_period):
                    found_old = True
                    break

                times = convert_utc_to_formats(row[2]["text"])
                details = row[5]["text"].splitlines()
                sold_amt, sold, bought_amt, bought = details[:4]
                sold_addr = row[5]["hrefs"][0].split("/")[-1] if row[5]["hrefs"] else None
                bought_addr = row[5]["hrefs"][1].split("/")[-1] if len(row[5]["hrefs"])>1 else None

                record = {
                    "signature": row[1]["text"],
                    "trade_time_utc": times["utc_time"],
                    "trade_time_relative": times["relative_time"],
                    "trade_time_unix": times["unix_timestamp"],
                    "token_sold": sold,
                    "token_sold_amount": sold_amt,
                    "token_sold_address": sold_addr,
                    "token_bought": bought,
                    "token_bought_amount": bought_amt,
                    "token_bought_address": bought_addr,
                    "txn_usd_value": row[6]["text"]
                }
                new_data.append(record)

            elapsed = time.time() - start
            logger.info(f"Page {pages_counter+1} done in {elapsed:.2f}s")

            if found_old:
                logger.info("Encountered old transactions; stopping early.")
                break

            if pages_counter < total_pages - 1:
                await page.locator("div:nth-child(2) > button:nth-child(4)").click()
                pages_counter += 1
                await page.wait_for_timeout(5000)
                if pages_counter % 5 == 0:
                    await check_and_solve_cloudflare(page, context)
            else:
                break

        await save_accumulated_data(address, new_data)
        return new_data

async def get_all_account_signatures(address, update, proxy=None, analyses_period=7, active_proxies=None):
    URL = "http://solscan.io"
    pages_counter = 0
    total_processing_time = 0
    
    # New data container (duplicates are not filtered)
    new_data = []
    found_old_txn = False  # Flag for transactions older than the threshold days

    try:
        async with async_playwright() as p:
            browser_options = {
                "headless": True,
                "slow_mo": 0,
            }
            browser = await p.firefox.launch(**{k: v for k, v in browser_options.items() if v is not None})
            context = await browser.new_context(ignore_https_errors=True)
            page = await context.new_page() 

            target = f"{URL}/account/{address}?activity_type=ACTIVITY_TOKEN_SWAP&activity_type=ACTIVITY_AGG_TOKEN_SWAP#defiactivities"
            await page.goto(target, timeout=90000)
            
            if not await check_and_solve_cloudflare(page, context):
                logger.error("Failed to bypass Cloudflare. Aborting.")
                return []

            logger.info(f"Accessed URL: {target}")

            # Wait for activities to load
            await page.wait_for_selector(".caption-bottom", timeout=60000)

            # Get total activities count
            activity_element = await page.get_by_text('activities(s)').first.text_content()
            match = re.search(r"Total\s([\d,]+)", activity_element)
            if not match:
                logger.warning("Total activities not found.")
                return
            
            total_activities = int(match.group(1).replace(",", ""))
            rows_per_page = 100
            total_pages = (total_activities + rows_per_page - 1) // rows_per_page
            logger.info(f"Total activities: {total_activities} | Pages: {total_pages}")
            
            # Determine if we should apply early stopping due to date filtering
            apply_early_stopping = total_pages > 10
            logger.info(f"Apply early stopping: {apply_early_stopping} (total pages: {total_pages}) | (Duration: {analyses_period})")

            cutoff_date = datetime.now() - timedelta(days=analyses_period)
            logger.info(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")

            # Set rows per page to 100
            pagination_button = page.locator("button[role='combobox']")
            await pagination_button.click()
            await page.get_by_label("100", exact=True).get_by_text("100").click()

            await page.wait_for_timeout(10000)

            time_column = page.locator("th:has-text('Time') .flex-row").first
            await time_column.click()
            await page.wait_for_timeout(2000)

            # Updated evaluation: return an object for each cell with both text and hrefs.
            while pages_counter < total_pages:
                page_start_time = time.time()
                logger.info(f"Processing page {pages_counter + 1} of {total_pages}")
                
                rows_data = await page.evaluate('''() => {
                    const rows = Array.from(document.querySelectorAll(".caption-bottom tr:not(:empty)"));
                    return rows.map(row => {
                        const cells = Array.from(row.querySelectorAll("td"));
                        return cells.map(cell => {
                            const cellText = cell.innerText.trim();
                            // Get all anchor hrefs (if any) in the cell.
                            const anchors = Array.from(cell.querySelectorAll("a"));
                            const hrefs = anchors.map(a => a.getAttribute("href"));
                            return { text: cellText, hrefs: hrefs };
                        });
                    });
                }''')
                logger.info(f"Row_data: {rows_data}")
                logger.info(f"Found {len(rows_data)} rows on page {pages_counter + 1}")
                processed_data = []

                for row in rows_data:
                    # Skip rows with fewer cells than expected.
                    if not row or len(row) < 7:
                        continue

                    # Extract basic fields using the cell object's text property.
                    signature = row[1]["text"]
                    trade_time_utc = row[2]["text"]
                    transaction_by = row[4]["text"]
                    
                    # Check if transaction is within the analyses_period threshold.
                    try:
                        if not is_within_n_days(trade_time_utc, analyses_period=analyses_period):
                            logger.info(f"Skipping transaction older than {analyses_period} days: {trade_time_utc}")
                            found_old_txn = True
                            continue
                    except Exception as e:
                        logger.warning(f"Error parsing relative time '{trade_time_utc}': {e}")
                        continue
                        
                    # Convert the UTC time to relative time and Unix timestamp.
                    time_formats = convert_utc_to_formats(trade_time_utc)
                   
                    # The token details cell (assumed at index 5) contains newline-separated data.
                    details_cell = row[5]
                    details_lines = details_cell["text"].split("\n")
                    if len(details_lines) < 4:
                        continue
                    token_sold_amount = details_lines[0].strip()
                    token_sold = details_lines[1].strip()
                    token_bought_amount = details_lines[2].strip()
                    token_bought = details_lines[3].strip()

                    token_sold_address = None
                    token_bought_address = None
                    
                    # Process the hrefs based on token names
                    if len(details_cell["hrefs"]) == 1:
                        # Only one href - need to determine if it belongs to token_sold or token_bought
                        if token_sold == 'SOL' or token_sold == 'WSOL':
                            # SOL is the sold token, so the href must be for the bought token
                            token_sold_address = 'So11111111111111111111111111111111111111112'
                            token_bought_address_str = details_cell["hrefs"][0]
                            token_bought_address = token_bought_address_str.replace('/token/', '') if token_bought_address_str else None
                        elif token_bought == 'SOL' or token_bought == 'WSOL':
                            # SOL is the bought token, so the href must be for the sold token
                            token_sold_address_str = details_cell["hrefs"][0]
                            token_sold_address = token_sold_address_str.replace('/token/', '') if token_sold_address_str else None
                            token_bought_address = 'So11111111111111111111111111111111111111112'
                        else:
                            # Neither is SOL, assume the href is for the token_sold
                            token_sold_address_str = details_cell["hrefs"][0]
                            token_sold_address = token_sold_address_str.replace('/token/', '') if token_sold_address_str else None
                    elif len(details_cell["hrefs"]) >= 2:
                        # Two or more hrefs - follow the original pattern
                        token_sold_address_str = details_cell["hrefs"][0]
                        token_bought_address_str = details_cell["hrefs"][1]
                        token_sold_address = token_sold_address_str.replace('/token/', '') if token_sold_address_str else None
                        token_bought_address = token_bought_address_str.replace('/token/', '') if token_bought_address_str else None

                    # Ensure SOL addresses are set correctly
                    if token_bought == 'SOL' and token_bought_address is None:
                        token_bought_address = 'So11111111111111111111111111111111111111112'
                    
                    if token_sold == 'SOL' and token_sold_address is None:
                        token_sold_address = 'So11111111111111111111111111111111111111112'
                        
                    # Also handle WSOL address correctly
                    if token_bought == 'WSOL' and token_bought_address is None:
                        token_bought_address = 'So11111111111111111111111111111111111111112'
                    
                    if token_sold == 'WSOL' and token_sold_address is None:
                        token_sold_address = 'So11111111111111111111111111111111111111112'

                    txn_usd_value = row[6]["text"]
                    record = {
                        # "account": address,
                        "signature": signature,
                        "trade_time_utc": time_formats["utc_time"],
                        "trade_time_relative": time_formats["relative_time"],
                        "trade_time_unix": time_formats["unix_timestamp"],
                        # "transaction_by": transaction_by,
                        "token_sold": token_sold,
                        "token_sold_amount": token_sold_amount,
                        "token_sold_address": token_sold_address,
                        "token_bought": token_bought,
                        "token_bought_amount": token_bought_amount,
                        "token_bought_address": token_bought_address,
                        "txn_usd_value": txn_usd_value,
                        # 'sol_amount_in_txn': None,
                        # 'sell_transfer_count': len(sell_transfers),
                        # 'buy_transfer_count': len(buy_transfers),
                        # 'gas_fee': feePaid,
                        # 'jito_fees': 0,
                    }
                    processed_data.append(record)

                new_data.extend(processed_data)

                page_end_time = time.time()
                page_duration = page_end_time - page_start_time
                total_processing_time += page_duration
                rows_per_second = len(rows_data) / page_duration if page_duration > 0 else 0
                logger.info(f"Page {pages_counter + 1} processed in {page_duration:.2f}s ({rows_per_second:.2f} rows/s)")
              
                if apply_early_stopping and found_old_txn:
                    logger.info(f"Found transactions older than {analyses_period} days. Stopping early.")
                    break

                # Check if there are more pages.
                next_button = page.locator("div:nth-child(2) > button:nth-child(4)")
                if await next_button.is_disabled() or pages_counter >= total_pages - 1:
                    logger.info("No more pages or reached the last page. Exiting pagination.")
                    break

                await next_button.click()
                pages_counter += 1
                logger.info(f"Moving to page {pages_counter + 1}")
                await page.wait_for_timeout(5000)
            
            combined_data = new_data
            
            # Optional final filtering (if not already using early stopping).
            if not apply_early_stopping:
                logger.info(f"Performing final filtering for transactions within {analyses_period} days")
                filtered_data = []
                for item in combined_data:
                    trade_time = item.get("trade_time", "")
                    try:
                        if is_within_n_days(trade_time, analyses_period=analyses_period):
                            filtered_data.append(item)
                    except Exception as e:
                        logger.warning(f"Error during final filtering for time '{trade_time}': {e}")
                        filtered_data.append(item)
                logger.info(f"Filtered data: {len(filtered_data)} out of {len(combined_data)} records remain")
                combined_data = filtered_data
            
            await save_accumulated_data(address, combined_data)
            
            pages_processed = pages_counter + 1
            avg_time_per_page = total_processing_time / pages_processed if pages_processed > 0 else 0
            logger.info(f"Scraping completed. Processed {pages_processed} pages in {total_processing_time:.2f}s (avg {avg_time_per_page:.2f}s per page)")
            logger.info(f"Total records after filtering: {len(combined_data)}")
            
            return combined_data

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        try:
            if 'page' in locals() and page and not page.is_closed():
                await take_error_screenshot(page, "pagination_error")
        except Exception as ss_err:
            logger.error(f"Could not take final error screenshot: {ss_err}")

async def get_all_signatures(mint_address, active_proxies=None, proxy=None):
    URL = "http://solscan.io"
    pages_counter = 0
    total_processing_time = 0

    metadata_dict_s = [mint_address]
    metadata_dict = await fetch_and_store_metadata(metadata_dict_s, proxy=None)
    # Ensure check_symbol is a string
    check_symbol = metadata_dict.get(mint_address, {})\
                                .get('content', {})\
                                .get('metadata', {})\
                                .get('symbol', 'N/A')
    
    # Load any existing data if available
    all_data = await load_existing_data(mint_address)
    
    # Create a set of existing signatures for quick lookup
    existing_signatures = set()
    for item in all_data:
        if item.get("signature") and item["signature"] != "N/A":
            existing_signatures.add(item["signature"])
    
    logger.info(f"Found {len(existing_signatures)} existing signatures")
    
    # New data container
    new_data = []
    found_existing = False

    try:
        async with async_playwright() as p:
            browser_options = {
                "headless": True,
                "slow_mo": 0,
            }

            browser = await p.firefox.launch(**{k: v for k, v in browser_options.items() if v is not None})
            context = await browser.new_context(ignore_https_errors=True)
            page = await context.new_page()

            target_url = f"{URL}/token/{mint_address}?activity_type=ACTIVITY_TOKEN_SWAP&activity_type=ACTIVITY_AGG_TOKEN_SWAP#defiactivities"
            await page.goto(target_url, timeout=90000)
            logger.info(f"Accessed URL: {target_url}")

            if not await check_and_solve_cloudflare(page, context):
                logger.error("Failed to bypass Cloudflare. Aborting.")
                return all_data


            # Wait for activities to load
            await page.wait_for_selector(".caption-bottom", timeout=60000)

            # Get total activities count
            activity_element = await page.get_by_text('activities(s)').first.text_content()
            match = re.search(r"Total\s([\d,]+)", activity_element)
            if not match:
                logger.warning("Total activities not found.")
                return
                
            total_activities = int(match.group(1).replace(",", ""))
            rows_per_page = 100
            total_pages = (total_activities + rows_per_page - 1) // rows_per_page
            logger.info(f"Total activities: {total_activities} | Pages: {total_pages}")

            # Set rows per page to 100
            pagination_button = page.locator("button[role='combobox']")
            await pagination_button.click()
            await page.get_by_label("100", exact=True).get_by_text("100").click()
            await page.wait_for_timeout(3000)

            while pages_counter < total_pages:
                page_start_time = time.time()
                logger.info(f"Processing page {pages_counter + 1} of {total_pages}")
                
                # Fetch all rows' data with a single evaluation in the browser context
                rows_data = await page.evaluate('''
                    () => {
                        const rows = Array.from(document.querySelectorAll(".caption-bottom tr:not(:empty)"));
                        return rows.map(row => {
                            const cells = Array.from(row.querySelectorAll("td")).map(td => td.innerText.trim());
                            return cells;
                        });
                    }
                ''')
                logger.info(f"Row_data: {rows_data}")
                logger.info(f"Found {len(rows_data)} rows on page {pages_counter + 1}")
                processed_data = []

                for row in rows_data:
                    # Skip any empty rows or rows that don't have the expected number of columns.
                    if not row or len(row) < 7:
                        continue

                    # Extract the values assuming a fixed column order.
                    signature = row[1].strip()
                    trade_time = row[2].strip()
                    transaction_by = row[4].strip()
                    
                    # The 6th element is a newline-separated string
                    details = row[5].splitlines()
                    if len(details) < 4:
                        continue

                    token_sold_amount = details[0].strip()
                    token_sold = details[1].strip()
                    token_bought_amount = details[2].strip()
                    token_bought = details[3].strip()

                    txn_usd_value = row[6].strip()

                    record = {
                        "mint_addr": mint_address,
                        "mint_symbol": check_symbol,
                        "signature": signature,
                        "trade_time": trade_time,
                        "transaction_by": transaction_by,
                        "token_sold_amount": token_sold_amount,
                        "token_sold": token_sold,
                        "token_bought_amount": token_bought_amount,
                        "token_bought": token_bought,
                        "txn_usd_value": txn_usd_value
                    }
                    
                    processed_data.append(record)

                # Check for duplicates by filtering out records with signatures that already exist
                page_new_data = []
                for item in processed_data:
                    sig = item.get("signature")
                    if sig and sig != "N/A":
                        if sig in existing_signatures:
                            # Duplicate found; skip this item.
                            continue
                        else:
                            page_new_data.append(item)
                            # Add signature to the set so that subsequent pages also consider it
                            existing_signatures.add(sig)
                    else:
                        # If signature is missing or invalid, include it or handle accordingly
                        page_new_data.append(item)
                
                if page_new_data:
                    logger.info(f"Added {len(page_new_data)} new records on page {pages_counter + 1}.")
                    new_data.extend(page_new_data)
                else:
                    logger.info(f"All records on page {pages_counter + 1} are duplicates.")
                    found_existing = True

                # Update combined data and save
                combined_data = new_data + all_data
                await save_accumulated_data(mint_address, combined_data)

                page_end_time = time.time()
                page_duration = page_end_time - page_start_time
                total_processing_time += page_duration
                rows_per_second = len(rows_data) / page_duration if page_duration > 0 else 0
                logger.info(f"Page {pages_counter + 1} processed in {page_duration:.2f}s ({rows_per_second:.2f} rows/s)")
                
                # Stop if only duplicates are found on the current page
                if found_existing and len(page_new_data) == 0:
                    logger.info("No new records found on this page. Stopping as existing data is caught up.")
                    break

                # Check if there are more pages
                next_button = page.locator("div:nth-child(2) > button:nth-child(4)")
                if await next_button.is_disabled() or pages_counter >= total_pages - 1:
                    logger.info("No more pages or reached the last page. Exiting pagination.")
                    break

                await next_button.click()
                pages_counter += 1
                logger.info(f"Moving to page {pages_counter + 1}")
                await page.wait_for_timeout(5000)
            
            # Final save to ensure everything is captured.
            combined_data = new_data + all_data
            await save_accumulated_data(mint_address, combined_data)
            
            pages_processed = pages_counter + 1
            avg_time_per_page = total_processing_time / pages_processed if pages_processed > 0 else 0
            logger.info(f"Scraping completed. Processed {pages_processed} pages in {total_processing_time:.2f}s (avg {avg_time_per_page:.2f}s per page)")
            logger.info(f"Added {len(new_data)} new records. Total records: {len(combined_data)}")
            
            return combined_data

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        # Try to take a final screenshot if browser is still available
    try:
        # Only attempt screenshot if the page is available and not closed
        if 'page' in locals() and page and not page.is_closed():
            await take_error_screenshot(page, "pagination_error")
    
    except Exception as ss_err:
        logger.error(f"Could not take final error screenshot: {ss_err}")

async def fetch_token_DEFI_transactions(update, active_proxies, proxy, mint_address):
    relevant_transactions = []
    
    # Fetch all transaction data
    all_transactions = await get_all_signatures(mint_address, active_proxies, proxy)
    
    # Filter relevant transactions if needed (or keep all)
    if all_transactions:
        relevant_transactions.extend(all_transactions)
    else:
        print(f"No relevant transactions found in this batch for {mint_address}.")
        
    return relevant_transactions

async def get_asset(sol_address, proxy):
    url = GET_ASSET_URL
    # If a proxy is provided, use it; otherwise, no proxy will be used
    connector = None
    if proxy:
        ip, port, username, password = proxy.split(':')
        proxy_url = f"http://{username}:{password}@{ip}:{port}"
        connector = TCPConnector()

    # Create the session with or without a proxy
    async with ClientSession(connector=connector) as session:
        async with session.post(
            url,
            headers={'Content-Type': 'application/json'},
            data=json.dumps({
                'jsonrpc': '2.0',
                'id': 'my-id',
                'method': 'getAsset',
                'params': {
                    'id': sol_address,
                    'displayOptions': {
                        'showFungible': True  # return details about a fungible token
                    }
                }
            }),
            proxy=proxy_url if proxy else None  # Add the proxy here if provided
        ) as response:
            result = await response.json()
            asset = result['result']
            token_info = asset['token_info']
            price_per_token = token_info['price_info']['price_per_token']
            return price_per_token

@celery.task(name='fetch_transactions')
def fetch_transactions_task(active_proxies, proxies, mint_address):
    logger.info(f"Starting fetch_transactions_task with mint_address: {mint_address}")
    try:
        # Create a new event loop to run the async function
        loop = new_event_loop()
        set_event_loop(loop)
        # Call your async function
        logger.info(f"About to call fetch_token_DEFI_transactions for {mint_address}")
        result = loop.run_until_complete(fetch_token_DEFI_transactions(None, active_proxies, proxies, mint_address))
        logger.info(f"fetch_token_DEFI_transactions completed for {mint_address}")
        loop.close()
        return result
    except Exception as e:
        import traceback
        logger.error(f"Task error in fetch_transactions_task: {e}")
        logger.error(traceback.format_exc())
        raise  # Re-raise to mark task as failed

@celery.task(name='fetch_account_transactions')
def fetch_account_transactions_task(address, update, proxies, analyses_period, active_proxies):
    logger.info(f"Starting fetch_account_transactions_task with address: {address}")
    try:
        # Create a new event loop to run the async function
        loop = new_event_loop()
        set_event_loop(loop)
        # Call your async function
        result = loop.run_until_complete(get_all_account_signatures(address, update, proxies, analyses_period, active_proxies))
        loop.close()
        return result
    except Exception as e:
        import traceback
        logger.error(f"Task error in fetch_account_transactions_task: {e}")
        logger.error(traceback.format_exc())
        raise  # Re-raise to mark task as failed

@celery.task(name='fetch_check_token_transactions')
def fetch_check_token_transactions_task(token_mint, active_proxies):
    logger.info(f"Starting fetch_check_token_transactions_task with tokens: {token_mint}")
    try:
        global proxy_stats, proxy_cooldowns
        proxy_stats = {}
        proxy_cooldowns = {}

        start_time = time.time()
        start_datetime = datetime.now()
        
        loop = new_event_loop()
        set_event_loop(loop)
        results = loop.run_until_complete(process_tokens_in_batches(token_mint, active_proxies))
        
        end_time = time.time()
        end_datetime = datetime.now()
        execution_time = end_time - start_time
        total_tokens = len(token_mint)
        successful_tokens = sum(1 for v in results.values() if v.get("score_normalised") != "N/A")
        success_rate = (successful_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        
        # Save overall debug information if needed
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"debug_{token_mint}_at_{timestamp}.json"
        try:
            debug_data = {
                "results": results,
                "proxy_stats": proxy_stats,
                "success_rate": f"{success_rate:.2f}%",
                "execution_time": f"{execution_time:.2f}s",
                "start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "end_time": end_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "cached_tokens": sum(1 for token in token_mint if get_cached_token_data(token) is not None)
            }
            with open(filename, "w") as f:
                json.dump(debug_data, f, indent=4)
            logger.info(f"Debug information saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving debug file: {e}")
        
        logger.info(f"Request completed. Success rate: {success_rate:.2f}% ({successful_tokens}/{total_tokens})")
        logger.info(f"Execution started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} and ended at {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        
        loop.close()
        
        return {
            "results": results,
            "stats": {
                "total_tokens": total_tokens,
                "successful_fetches": successful_tokens,
                "success_rate": f"{success_rate:.2f}%",
                "execution_time": f"{execution_time:.2f}s",
                "start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "end_time": end_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "cached_tokens_used": sum(1 for token in token_mint if token in results and "from_cache" in results[token])
            }
        }
    
    except Exception as e:
        import traceback
        logger.error(f"Task error in fetch_check_token_transactions_task: {e}")
        logger.error(traceback.format_exc())
        raise

# Flask API endpoint to check multiple proxies
@app.route("/checkProxies", methods=["POST"])
def check_proxies():
    data = request.get_json()
    proxies = data.get("proxies", [])

    if not proxies:
        return jsonify({"error": "Proxies list is required"}), 400

    active_proxies = process_proxies_in_batches(proxies)
    return jsonify({"active_proxies": active_proxies})

@app.route("/getSolBalance", methods=["GET"])
def get_sol_balance():
    if not verify_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    # Extract address and proxy from query params
    address = request.args.get("address")
    proxy = request.args.get("proxy")

    if not address:
        return jsonify({"error": "Missing address parameter"}), 400

    # JSON-RPC payload for Solana balance check
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBalance",
        "params": [address, {"commitment": "confirmed"}]
    }

    # Setup proxies if provided
    proxies = None
    if proxy:
        try:
            ip, port, username, password = proxy.split(':')
            proxy_url = f"http://{username}:{password}@{ip}:{port}"
            proxies = {
                "http": proxy_url,
                "https": proxy_url
            }
        except ValueError:
            return jsonify({"error": "Invalid proxy format"}), 400

    # Send the request to Solana RPC
    response = requests.post(GET_NATIVE_BAL_URL, json=payload, proxies=proxies)

    if response.status_code == 200:
        result = response.json()
        balance = result.get("result", {}).get("value", 0) / 1_000_000_000  # Convert lamports to SOL
        return jsonify({"balance": balance})
    else:
        return jsonify({"error": f"RPC Error: {response.status_code}", "details": response.text}), 500

@app.route('/get_asset', methods=['POST'])
async def fetch_asset():
    data = request.json
    sol_address = data.get("sol_address")
    proxy = data.get("proxy", None)  # Optional proxy

    if not sol_address:
        return jsonify({"error": "Solana address is required"}), 400

    try:
        price_per_token = await get_asset(sol_address, proxy)
        return jsonify({"price_per_token": price_per_token})  # Ensure response is JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return error message as JSON


@app.route('/fetchAccountTrans', methods=['POST'])
def fetch_account_transactions_fetcher():
    data = request.json
    address = data.get('address')
    update = data.get('update')
    proxies = data.get('proxies')
    analyses_period = data.get('analyses_period')
    active_proxies = data.get('active_proxies')


    if not address or not analyses_period or not active_proxies:
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Enqueue the background task and immediately return the task id
    task = fetch_account_transactions_task.delay(address, update, proxies, analyses_period, active_proxies)
    logger.info(f"Created task with ID: {task.id}")
    return jsonify({"task_id": task.id}), 202

@app.route('/fetchTokenDEFITrans', methods=['POST'])
def fetch_token_transactions():
    data = request.json
    logger.info(f"Received request data: {data}")
    active_proxies = data.get('active_proxies')
    proxies = data.get('proxies')
    mint_address = data.get('mint_address')

    if not mint_address or not active_proxies:
        logger.error("Missing required parameters")
        return jsonify({"error": "Missing required parameters"}), 400

    # Enqueue the background task and immediately return the task id
    task = fetch_transactions_task.delay(active_proxies, proxies, mint_address)
    logger.info(f"Created task with ID: {task.id}")
    return jsonify({"task_id": task.id}), 202


@app.route("/checkScamTokens", methods=["POST"])
def check_tokens():
    """
    Expects a JSON payload with 'tokens' (list) and 'proxies' (list).
    Processes tokens while utilizing optimized caching system.
    """
    data = request.get_json()
    token_mint = data.get("tokens", [])
    active_proxies = data.get("proxies", [])
    
    if not token_mint or not active_proxies:
        return jsonify({"error": "Missing tokens or proxies"}), 400
    
    # Enqueue the background task and immediately return the task id
    task = fetch_check_token_transactions_task.delay(token_mint, active_proxies)
    logger.info(f"Created task with ID: {task.id}")
    return jsonify({"task_id": task.id}), 202

@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    task_result = fetch_transactions_task.AsyncResult(task_id) or fetch_account_transactions_task.AsyncResult(task_id) or fetch_account_transactions_fetcher.AsyncResult(task_id)
    if task_result is None:
        return jsonify({"error": "Task not found"}), 404
    
    if task_result.state == 'PENDING':
        response = {'state': task_result.state, 'status': 'Pending...'}
    elif task_result.state == 'SUCCESS':
        if isinstance(task_result.result, dict) and 'error' in task_result.result:
            # Task completed but returned an error
            response = {
                'state': 'FAILURE',
                'error': task_result.result.get('error'),
                'traceback': task_result.result.get('traceback')
            }
        else:
            response = {'state': task_result.state, 'result': task_result.result}
    elif task_result.state == 'FAILURE':
        response = {
            'state': task_result.state, 
            'error': str(task_result.result),
            'traceback': task_result.traceback
        }
    else:
        response = {'state': task_result.state, 'status': 'Processing...'}
    
    logger.info(f"Task {task_id} status: {task_result.state}")
    return jsonify(response)

@app.route('/fetchMetadata', methods=['POST'])
async def fetch_the_metadata():
    data = request.get_json()
    unique_token_addresses = data.get("unique_token_addresses", [])
    proxy = data.get("proxy", None)  # Optional proxy

    if not unique_token_addresses:
        return jsonify({"error": "No token addresses provided"}), 400

    try:
        metadata = await fetch_and_store_metadata(unique_token_addresses, proxy)
        return jsonify({"metadata": metadata})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
