# File 1
import requests
from httpx import AsyncClient, HTTPStatusError, TimeoutException, RequestError
import random
from collections import defaultdict
import time
import aiohttp
import logging
import os
from os import getenv, path, makedirs
from playwright.async_api import async_playwright
import re
import redis.asyncio as aioredis
from asyncio import TimeoutError, create_task, sleep, run, Semaphore, gather, new_event_loop, set_event_loop
from aiohttp import ClientSession, TCPConnector, ClientTimeout
from aiohttp.client_exceptions import ClientError
from redis.asyncio import Redis, ConnectionPool
import json
from collections import deque
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, Timeout, RequestException
# from datetime import datetime, timedelta
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from asyncio import TimeoutError, sleep, run, Semaphore, gather, new_event_loop, set_event_loop
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

load_dotenv()
API_KEY = getenv('HELIUS_API_KEY')
BAGSCAN_API_KEY = getenv('BAGSCAN_API_KEY')
SERVER_URL = getenv('SERVER_URL')

TELEGRAM_BOT_TOKEN = getenv('TELEGRAM_BOT_TOKEN')

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
GET_ASSET_URL = f'https://mainnet.helius-rpc.com/?api-key={API_KEY}'

MAX_RETRIES = 5
SLEEP_TIME = 30
semaphore = Semaphore(10)

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
# proxy_usage = {}
proxy_usage = defaultdict(lambda: Semaphore(1))  # Ensure all proxies have a semaphore

def get_sol_balance(address, proxy):
    """
    Fetch the SOL balance for a given address, optionally using a proxy.
    """
    params = {"address": address}
    if proxy:
        params["proxy"] = proxy

    response = requests.get(
        f"{SERVER_URL}/getSolBalance",
        headers={"Authorization": f"Bearer {BAGSCAN_API_KEY}"},
        params=params
    )

    if response.status_code == 200:
        return response.json()  # Returns {"balance": <amount>}
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Function to load proxies from a file
async def load_proxies(file_path):
    async with aiofiles.open(file_path, 'r') as file:
        proxies = [line.strip() for line in await file.readlines()]
    return proxies

async def get_next_available_proxy(proxy_cycle):
    """Get the next available proxy that is not fully occupied."""
    while True:
        proxy = next(proxy_cycle)
        if proxy not in proxy_usage:
            proxy_usage[proxy] = Semaphore(PROXY_LIMIT)
        if not proxy_usage[proxy].locked():  # Ensure proxy is not overloaded
            return proxy

# Create Redis connection
async def create_redis_connection_pool():
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            r = await aioredis.from_url(
                'redis://redis-16610.c14.us-east-1-2.ec2.redns.redis-cloud.com:16610',
                password='ByR0uxTKKJ8YqUL0Rf4sdyfsWjIsXAAr'
            )
            # r = Redis(connection_pool=pool)
            print("Connected to Redis successfully.")
            return r
        except (aioredis.ConnectionError, aioredis.TimeoutError) as e:
            attempt += 1
            print(f"Connection attempt {attempt} failed: {e}. Retrying in 1 second...")
            await sleep(SLEEP_TIME)

    print("Failed to connect to Redis after multiple attempts.")
    raise Exception("Unable to connect to Redis.")

def parse_relative_time(relative_time_str, current_time=None):
    """Convert a relative time string like '5 mins ago' to a timestamp."""
    if current_time is None:
        current_time = datetime.now()
        
    try:
        # Handle different time formats
        if "ago" in relative_time_str:
            parts = relative_time_str.lower().replace("ago", "").strip().split()
            
            if len(parts) >= 2:  # Make sure we have both number and unit
                amount = int(parts[0])
                unit = parts[1].rstrip('s')  # Remove trailing 's' if present
                
                if unit == "min" or unit == "minute":
                    return current_time - timedelta(minutes=amount)
                elif unit == "hour":
                    return current_time - timedelta(hours=amount)
                elif unit == "day":
                    return current_time - timedelta(days=amount)
                elif unit == "second":
                    return current_time - timedelta(seconds=amount)
        
        # If format not recognized or parsing failed, log it and return current time
        logger.warning(f"Could not parse relative time: {relative_time_str}")
        return current_time
    except Exception as e:
        logger.error(f"Error parsing relative time '{relative_time_str}': {e}")
        return current_time
    
# Function to distribute proxies to addresses using round-robin
def round_robin_proxies(proxies, addresses):
    proxy_map = {}
    proxy_iter = iter(proxies)  # Create an iterator from the list of proxies
    
    for address in addresses:
        proxy_map[address] = next(proxy_iter)  # Assign the next proxy in the round-robin cycle
        
        # If we've run out of proxies, restart the cycle
        if proxy_iter is None:
            proxy_iter = iter(proxies)
    
    return proxy_map

#For main file 
#For main file 
#For main file
# Extract unique tokens from transactions
def extract_unique_tokens(relevant_transactions):
    unique_tokens = set()

    # Ensure we're getting the actual list
    # transactions_list = relevant_transactions

    for transaction in relevant_transactions:
        token_sold = transaction.get("token_sold_address")
        token_bought = transaction.get("token_bought_address")

        if token_sold:
            unique_tokens.add(token_sold)
        if token_bought:
            unique_tokens.add(token_bought)

    unique_tokens_list = list(unique_tokens)
    
    # Debugging output
    print("Extracted Unique Tokens:", unique_tokens_list)

    return unique_tokens_list


def generate_wallet_summary(sol_balance_value, number_of_tokens_traded, profitable_trades,
        unprofitable_trades, winrate, avg_hld_tme, average_buy_per_trade, average_trades_per_day,
        first_transaction_trade_time, average_buy_size, total_sol_spent_usd, total_sol_spent,
        total_sol_made_usd, total_sol_made, profit_loss, profit_loss_usd, profit_loss_roi, tokens_with_more_sells_than_buys, 
        tokens_with_short_hold_duration, tokens_with_high_scam_score, longest_winning_streak, longest_losing_streak, total_gas_fees, average_gas_fee,
        total_jito_fees, average_jito_fee, address, analyses_period, mc_categories):
    summary = []
    summary.append(f"🧾 Overall Wallet Summary for {address} in {analyses_period} days")
    summary.append("=" * 30)
    if sol_balance_value is not None:
        summary.append(f"🔹 Balance: {sol_balance_value:.9f} SOL")
    summary.append(f"🔹 Number of Tokens Traded: {number_of_tokens_traded} tokens")
    summary.append(f"🔹 Profitable Trades: {profitable_trades} / {number_of_tokens_traded} tokens")
    summary.append(f"🔹 Unprofitable Trades: {unprofitable_trades} / {number_of_tokens_traded} tokens")
    summary.append(f"🔹 Winrate: {winrate:.2f}%\n")
    summary.append(f"🔹 Average Held Duration: {avg_hld_tme}")
    summary.append(f"🔹 Average Buys per Trade: {average_buy_per_trade:.2f}")
    summary.append(f"🔹 Average Trades per Day: {average_trades_per_day:.2f}")
    summary.append(f"🔹 Last Trade Time: {first_transaction_trade_time}\n")
    summary.append(f"🔹 Avg Buy Size (SOL): {average_buy_size:.2f}")
    # summary.append(f"🔹 Total Gas Fees (SOL): { total_gas_fees:.4f}")
    # summary.append(f"🔹 Avg Gas Fees (SOL): {average_gas_fee:.4f}")
    # summary.append(f"🔹 Total Jito Fees (SOL): { total_jito_fees:.4f}")
    # summary.append(f"🔹 Avg Jito Fees (SOL): {average_jito_fee:.4f}")
    summary.append(f"🔹 Total SOL Spent (USD): {total_sol_spent:.9f}SOL (${total_sol_spent_usd:.2f})")
    summary.append(f"🔹 Total SOL Made (USD): {total_sol_made:.9f}SOL (${total_sol_made_usd:.2f})")
    summary.append(f"🔹 PNL: {profit_loss:.9f} SOL (${profit_loss_usd:.2f})")
    summary.append(f"🔹 PNL ROI: {profit_loss_roi:.2f}%\n")
    
    summary.append(f"🔹 Tokens with More Sells than Buys: {tokens_with_more_sells_than_buys}")
    summary.append(f"🔹 Tokens with Short Hold Duration (< 1 min): {tokens_with_short_hold_duration}")
    summary.append(f"🔹 Scam tokens purchased: {tokens_with_high_scam_score}")
    summary.append("🔹 Longest Winning Streak: {}".format(longest_winning_streak))
    summary.append("🔹 Longest Losing Streak: {}".format(longest_losing_streak))
    
     # Market Capitalization Distribution
    summary.append("\n🔹 Market Capitalization Distribution of Tokens:")
    summary.append("=" * 30)
    for category, count in mc_categories.items():
        summary.append(f"{category}: {count} tokens")
    return "\n".join(summary)

def generate_best_trade(best_trade):
    if best_trade:
        trade_info = []
        trade_info.append("🏆 Best Trade Based on ROI")
        trade_info.append("=" * 30)
        trade_info.append(f"🔹 Token Address: {best_trade['token_address']} ({best_trade['symbol']})")
        trade_info.append(f"🔹 Buys: {best_trade['buys']}, Sells: {best_trade['sells']}")
        trade_info.append(f"🔹 Total Bought Amount: {best_trade['buy_amount']:.2f} {best_trade['symbol']} (Spent {best_trade['sol_spent']:.9f} SOL)")
        trade_info.append(f"🔹 Total Sold Amount: {best_trade['sell_amount']:.2f} {best_trade['symbol']} (Made {best_trade['sol_made']:.9f} SOL)")
        trade_info.append(f"🔹 Held Duration: {best_trade['hold_duration']}")
        trade_info.append(f"🔹 Profit: {best_trade['profit_loss']:.9f} SOL (${best_trade['profit_loss_usd']:.2f})")
        trade_info.append(f"🔹 ROI: {best_trade['pnl_roi']:.2f}%")
        return "\n".join(trade_info)
    else:
        return "No Best Trade!!"

def safe_int_conversion(value):
    try:
        return int(value)
    except ValueError:
        return 0

async def save_account_transactions_to_file(update, address, relevant_transactions, analyses_period, proxy, results):
    # print("Here")
    token_addresses = set()
    token_trade_summary = {}
    
    # Define the MC categories counters
    mc_categories = {
        '<5K': 0,
        '5K-30K': 0,
        '30K-100K': 0,
        '100K-300K': 0,
        '300K+': 0,
    }
    
    total_sol_spent = 0
    total_sol_made = 0
    profitable_trades = 0
    unprofitable_trades = 0
    number_of_tokens_traded = 0 
    best_trade = None
    best_trade_roi = -float('inf')
    
    # Initialize variables for tracking streaks
    current_winning_streak = 0
    current_losing_streak = 0
    longest_winning_streak = 0
    longest_losing_streak = 0
    total_gas_fees = 0.0
    total_jito_fees = 0.0
    transaction_count = 0
    tokens_with_high_scam_score = 0

    for transaction in relevant_transactions:
        token_sold = transaction.get("token_sold")
        token_bought = transaction.get("token_bought")

        token_addresses.add(token_sold)
        token_addresses.add(token_bought)

    # token_metadata = await fetch_the_metadata(token_addresses, proxy)
    sol_balance = get_sol_balance(address, proxy)
    logger.info(f"sol_balance: {sol_balance['balance']}")
    # sol_metadata = token_metadata.get(SOL_ADDR, {})
    # sol_cur_price = sol_metadata.get('token_info', {}).get('price_info', {}).get('price_per_token', 236)
    sol_cur_price = await get_asset(SOL_ADDR, proxy)
    logger.info(f"sol_cur_price: {sol_cur_price["price_per_token"]}")
    
    # Create a new Excel workbook and worksheet
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"Trade - {address[:10]}"

    # Track the current row
    current_row = 1

    # Add title and header rows
    ws.merge_cells('A1:N1')
    ws['A1'] = f"{analyses_period} Days Trade Analysis of Wallet {address}"
    ws['A1'].font = Font(size=16, bold=True)
    # ws['A1'].alignment = Alignment(horizontal="center")
    ws['A1'].alignment = Alignment(horizontal="center", vertical="center")

    # Increment the row for the header
    current_row += 1
    
    # Add column headers
    # headers = ["Signature", "Token Bought", "Token Bought Amount", "Token Sold", "Token Sold Amount", "Trade Time", "Sol Amount in Txn (SOL)", "USD Value", "Gas spent (SOL)", "Jito Fees (SOL)"]
    headers = ["Signature", "Token Bought", "Token Bought Amount", "Token Sold", "Token Sold Amount", "Trade Time", "USD Value"]
    ws.append(headers)
    
    # Style headers (bold, fill color)
    header_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    for cell in ws[2]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    # Increment the row after headers are filled
    current_row += 1
    # print("Here 3")

    for transaction in relevant_transactions:
        timestamp_rltv = transaction.get("trade_time_relative")
        trade_time_unix = transaction.get("trade_time_unix")
        # trade_time_utc = transaction.get("trade_time_utc")
        token_sold = transaction.get("token_sold")
        token_sold_address = transaction.get("token_sold_address")
        token_bought = transaction.get("token_bought")
        token_bought_address = transaction.get("token_bought_address")
        token_bought_amount_str = transaction.get("token_bought_amount")
        token_sold_amount_str = transaction.get("token_sold_amount")


        txn_usd_value_chr = transaction.get("txn_usd_value", "$0")
        usd_value_str = float(txn_usd_value_chr.replace("$", "").replace(",", "")) if txn_usd_value_chr and txn_usd_value_chr.lower() != "n/a" else 0.0

        token_bought_amount_int = token_bought_amount_str.replace(",", "") if token_bought_amount_str else 0
        token_sold_amount_int = token_sold_amount_str.replace(",", "") if token_sold_amount_str else 0
        
        # txn_usd_value = float(transaction.get('txn_usd_value', 0) or 0)
        def safe_float(value):
            try:
                # Handle if value is already a number or a string number.
                return float(value)
            except (ValueError, TypeError):
                return 0.0

        token_sold_amount = safe_float(token_sold_amount_int)
        token_bought_amount = safe_float(token_bought_amount_int)
        # usd_value = safe_float(transaction.get('txn_usd_value', 0))
        usd_value = safe_float(usd_value_str)

        gas_fee = float(transaction.get('gas_fee', 0) or 0)
        total_gas_fees += gas_fee
        jito_fee = float(transaction.get('jito_fees', 0) or 0)
        sol_amount_in_txn = float(transaction.get('sol_amount_in_txn', 0) or 0)
        total_jito_fees += jito_fee
        transaction_count += 1
        
        # Ensure that the values are not None, and replace None with a default value (e.g., 0.0)
        gas_fee = transaction.get('gas_fee', 0.0) if transaction.get('gas_fee') is not None else 0.0
        jito_fee = transaction.get('jito_fee', 0.0) if transaction.get('jito_fee') is not None else 0.0
       
        # Format the values to 9 decimal places, ensuring no scientific notation
        formatted_gas_fee = f"{gas_fee:.9f}"
        formatted_jito_fee = f"{jito_fee:.9f}"
        
        # Append transaction data to Excel sheet
        ws.append([
            transaction['signature'],
            # transaction['block_timestamp'],
            f"{token_bought_address} ({token_bought})",
            f"{token_bought_amount} {token_bought}",
            f"{token_sold_address} ({token_sold})",
            f"{token_sold_amount} {token_sold}",
            timestamp_rltv,
            # f"{sol_amount_in_txn}",
            usd_value,
            # formatted_gas_fee,
            # formatted_jito_fee,
        ])
    
        # Formatting: alternate row shading
        if current_row % 2 == 0:
            for cell in ws[current_row]:
                cell.fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
                cell.alignment = Alignment(horizontal="center")

        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=current_row, column=col)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            
        
        current_row += 1

        if token_sold_address not in token_trade_summary:
            token_trade_summary[token_sold_address] = {'buys': 0, 'sells': 0, 'total_buy_amount': 0, 'total_sell_amount': 0, 'sol_spent': 0, 'sol_made': 0, 'first_buy_timestamp': None, 'last_sell_timestamp': None, 'first_buy_mc': None, 'last_sell_mc': None, 'current_mc': None}
        if token_bought_address not in token_trade_summary:
            token_trade_summary[token_bought_address] = {'buys': 0, 'sells': 0, 'total_buy_amount': 0, 'total_sell_amount': 0, 'sol_spent': 0, 'sol_made': 0, 'first_buy_timestamp': None, 'last_sell_timestamp': None, 'first_buy_mc': None, 'last_sell_mc': None, 'current_mc': None}

        if token_sold_address == SOL_ADDR and token_bought_address != SOL_ADDR:
            if token_bought_address in results["results"]:
                token_data = results["results"][token_bought_address]
                
                token_trade_summary[token_bought_address]['buys'] += 1
                token_trade_summary[token_bought_address]['total_buy_amount'] += token_bought_amount
                token_trade_summary[token_bought_address]['sol_spent'] += token_sold_amount
                total_sol_spent += token_sold_amount
                token_trade_summary[token_bought_address]['first_buy_timestamp'] = trade_time_unix
                token_value = token_sold_amount / token_bought_amount if token_bought_amount else 0
                
                token_base_price = token_data.get("basePrice", "N/A")
                token_quote_price = token_data.get("quotePrice", "N/A")
                if token_base_price > token_quote_price:
                    token_bought_cur_price = token_quote_price
                else:
                    token_bought_cur_price = token_base_price

                if token_bought_cur_price == "N/A":
                    token_bought_cur_price = 0
                else:
                    token_bought_cur_price = float(token_bought_cur_price)
                
                token_bought_supply = token_data.get("supply", 0)
                if token_bought_supply == "N/A":
                    token_bought_supply = 0
                else:
                    token_bought_supply = float(token_bought_supply)
                
                token_bought_decimals = token_data.get("decimals", 0)
                if token_bought_decimals == "N/A":
                    token_bought_decimals = 0
                else:
                    token_bought_decimals = int(token_bought_decimals)
                
                # Calculate adjusted supply
                if token_bought_supply and token_bought_decimals:
                    token_bought_supply = token_bought_supply / (10 ** token_bought_decimals)
                
                first_buy_mc = token_value * sol_cur_price["price_per_token"] * token_bought_supply
                token_trade_summary[token_bought_address]['first_buy_mc'] = first_buy_mc
                    
                token_current_mc = token_bought_cur_price * token_bought_supply
                token_trade_summary[token_bought_address]['current_mc'] = token_current_mc
            
            logger.info(f"case 1")
            logger.info(f" Token Sold: token_trade_summary[{token_sold_address}]: {token_trade_summary[token_sold_address]}")
            logger.info(f" Token Bought: token_trade_summary[{token_bought_address}]: {token_trade_summary[token_bought_address]}")

            
        elif token_bought_address == SOL_ADDR and token_sold_address != SOL_ADDR:
            token_trade_summary[token_sold_address]['sells'] += 1
            token_trade_summary[token_sold_address]['total_sell_amount'] += token_sold_amount
            token_trade_summary[token_sold_address]['sol_made'] += token_bought_amount
            total_sol_made += token_bought_amount

            if token_trade_summary[token_sold_address]['last_sell_timestamp'] is None:
                token_trade_summary[token_sold_address]['last_sell_timestamp'] = trade_time_unix
                
                token_value = token_bought_amount / token_sold_amount if token_sold_amount else 0
                
                # Get token data from results
                if token_sold_address in results["results"]:
                    token_data = results["results"][token_sold_address]
                    
                    # Extract current price
                    token_base_price = token_data.get("basePrice", "N/A")
                    token_quote_price = token_data.get("quotePrice", "N/A")
                    if token_base_price > token_quote_price:
                        token_sold_cur_price = token_quote_price
                    else:
                        token_sold_cur_price = token_base_price

                    if token_sold_cur_price == "N/A":
                        token_sold_cur_price = 0
                    else:
                        token_sold_cur_price = float(token_sold_cur_price)
                    
                    # Extract supply
                    token_sold_supply = token_data.get("supply", "N/A")
                    if token_sold_supply == "N/A":
                        token_sold_supply = 0
                    else:
                        token_sold_supply = float(token_sold_supply)
                    
                    # Extract decimals
                    token_sold_decimals = token_data.get("decimals", "N/A")
                    if token_sold_decimals == "N/A":
                        token_sold_decimals = 0
                    else:
                        token_sold_decimals = int(token_sold_decimals)
                    
                    # Calculate adjusted supply
                    if token_sold_supply and token_sold_decimals:
                        token_sold_supply = token_sold_supply / (10 ** token_sold_decimals)
                    
                    # Calculate market caps
                    last_sell_mc = token_value * sol_cur_price["price_per_token"] * token_sold_supply
                    token_trade_summary[token_sold_address]['last_sell_mc'] = last_sell_mc
                    
                    token_current_mc = token_sold_cur_price * token_sold_supply
                    token_trade_summary[token_sold_address]['current_mc'] = token_current_mc
                else:
                    # Handle case where token data is not available
                    logger.warning(f"Token data not available for: {token_sold}")
                    token_trade_summary[token_sold_address]['last_sell_mc'] = None
                    token_trade_summary[token_sold_address]['current_mc'] = None
            logger.info(f"case 2")
            logger.info(f" Token Sold: token_trade_summary[{token_sold_address}]: {token_trade_summary[token_sold_address]}")
            logger.info(f" Token Bought: token_trade_summary[{token_bought_address}]: {token_trade_summary[token_bought_address]}")


        elif token_sold_address != SOL_ADDR and token_bought_address != SOL_ADDR:

            txn_usd_value_chr = transaction.get("txn_usd_value", "$0")
            usd_value_str = float(txn_usd_value_chr.replace("$", "").replace(",", "")) if txn_usd_value_chr and txn_usd_value_chr.lower() != "n/a" else 0.0
 
            def safe_float(value):
                try:
                    # Handle if value is already a number or a string number.
                    return float(value)
                except (ValueError, TypeError):
                    return 0.0

            txn_usd_value = safe_float(usd_value_str)

            if sol_cur_price['price_per_token'] and txn_usd_value:
                sol_amount = txn_usd_value / sol_cur_price['price_per_token']
            token_trade_summary[token_sold_address]['sells'] += 1
            token_trade_summary[token_bought_address]['buys'] += 1
            token_trade_summary[token_sold_address]['total_sell_amount'] += token_sold_amount
            token_trade_summary[token_bought_address]['total_buy_amount'] += token_bought_amount

            token_trade_summary[token_bought_address]['first_buy_timestamp'] = trade_time_unix
            token_value = txn_usd_value / token_bought_amount if token_bought_amount else 0

            # Get token_bought data from results instead of token_metadata
            if token_bought_address in results["results"]:
                token_data_bought = results["results"][token_bought_address]

                token_base_price = token_data_bought.get("basePrice", "N/A")
                token_quote_price = token_data_bought.get("quotePrice", "N/A")
                if token_base_price > token_quote_price:
                    token_bought_cur_price = token_quote_price
                else:
                    token_bought_cur_price = token_base_price

                if token_bought_cur_price == "N/A":
                    token_bought_cur_price = 0
                else:
                    token_bought_cur_price = float(token_bought_cur_price)

                # Extract supply
                token_bought_supply = token_data_bought.get("supply", 0)
                if token_bought_supply == "N/A":
                    token_bought_supply = 0
                else:
                    token_bought_supply = float(token_bought_supply)

                # Extract decimals and calculate adjusted supply
                token_bought_decimals = token_data_bought.get("decimals", 0)
                if token_bought_decimals == "N/A":
                    token_bought_decimals = 0
                else:
                    token_bought_decimals = int(token_bought_decimals)

                if token_bought_supply and token_bought_decimals:
                    token_bought_supply = token_bought_supply / (10 ** token_bought_decimals)
            else:
                token_bought_cur_price = 0
                token_bought_supply = 0

            # Compute first buy market cap and current market cap
            first_buy_mc = token_value * token_bought_supply
            token_trade_summary[token_bought_address]['first_buy_mc'] = first_buy_mc

            token_current_mc = token_bought_cur_price * token_bought_supply
            token_trade_summary[token_bought_address]['current_mc'] = token_current_mc

            # Process token_sold data if this is the first sell
            if token_trade_summary[token_sold_address]['last_sell_timestamp'] is None:

                txn_usd_value_chr = transaction.get("txn_usd_value", "$0")
                usd_value_str = float(txn_usd_value_chr.replace("$", "").replace(",", "")) if txn_usd_value_chr and txn_usd_value_chr.lower() != "n/a" else 0.0
    
                def safe_float(value):
                    try:
                        # Handle if value is already a number or a string number.
                        return float(value)
                    except (ValueError, TypeError):
                        return 0.0

                txn_usd_value = safe_float(usd_value_str)

                # txn_usd_value = float(transaction.get('txn_usd_value', 0) or 0)
                if sol_cur_price['price_per_token'] and txn_usd_value:
                    sol_amount = txn_usd_value / sol_cur_price['price_per_token']

                token_trade_summary[token_sold_address]['last_sell_timestamp'] = trade_time_unix
                token_value = txn_usd_value / token_sold_amount if token_sold_amount else 0

                if token_sold_address in results["results"]:
                    token_data_sold = results["results"][token_sold_address]

                    token_base_price = token_data_sold.get("basePrice", "N/A")
                    token_quote_price = token_data_sold.get("quotePrice", "N/A")
                    if token_base_price > token_quote_price:
                        token_sold_cur_price = token_quote_price
                    else:
                        token_sold_cur_price = token_base_price

                    if token_sold_cur_price == "N/A":
                        token_sold_cur_price = 0
                    else:
                        token_sold_cur_price = float(token_sold_cur_price)

                    # Extract supply
                    token_sold_supply = token_data_sold.get("supply", 0)
                    if token_sold_supply == "N/A":
                        token_sold_supply = 0
                    else:
                        token_sold_supply = float(token_sold_supply)

                    # Extract decimals and calculate adjusted supply
                    token_sold_decimals = token_data_sold.get("decimals", 0)
                    if token_sold_decimals == "N/A":
                        token_sold_decimals = 0
                    else:
                        token_sold_decimals = int(token_sold_decimals)

                    if token_sold_supply and token_sold_decimals:
                        token_sold_supply = token_sold_supply / (10 ** token_sold_decimals)
                else:
                    token_sold_cur_price = 0
                    token_sold_supply = 0

                last_sell_mc = token_value * token_sold_supply
                token_trade_summary[token_sold_address]['last_sell_mc'] = last_sell_mc

                token_current_mc = token_sold_cur_price * token_sold_supply
                token_trade_summary[token_sold_address]['current_mc'] = token_current_mc

            logger.info(f"case 3")
            logger.info(f" Token Sold: token_trade_summary[{token_sold_address}]: {token_trade_summary[token_sold_address]}")
            logger.info(f" Token Bought: token_trade_summary[{token_bought_address}]: {token_trade_summary[token_bought_address]}")


            token_trade_summary[token_sold_address]['sol_made'] += sol_amount
            token_trade_summary[token_bought_address]['sol_spent'] += sol_amount
            total_sol_made += sol_amount
            total_sol_spent += sol_amount
            # print("Here 4")
    
    total_sol_spent_usd = total_sol_spent * sol_cur_price["price_per_token"]
    total_sol_made_usd = total_sol_made * sol_cur_price["price_per_token"]
    profit_loss_usd = total_sol_made_usd - total_sol_spent_usd
    profit_loss_roi = (profit_loss_usd / total_sol_spent_usd) * 100 if total_sol_spent_usd else 0

    first_transaction_trade_time = relevant_transactions[0]['trade_time_relative'] if relevant_transactions else "N/A"
    total_buy_counts = 0 
    average_gas_fee = total_gas_fees / transaction_count if transaction_count else 0
    average_jito_fee = total_jito_fees / transaction_count if transaction_count else 0
    # print("Here 4a")
    

    # Append an empty row
    ws.append([])
    # Move to the next row
    current_row += 1
    # Add title and header rows
    ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=14)
    ws.cell(row=current_row, column=1, value="Token Trade Summary (Buys, Sells, Buy Amount, Sell Amount)").font = Font(size=10, bold=True)
    ws.cell(row=current_row, column=1).alignment = Alignment(horizontal="center", vertical="center")
    # Move to the next row
    current_row += 1

    tokens_with_more_sells_than_buys = 0
    tokens_with_short_hold_duration = 0
    total_held_duration_seconds = 0
    # print("Here 5")
    
    token_trade_list = []
    for token_address, counts in token_trade_summary.items():
        if counts['buys'] == 0 and counts['sells'] == 0:
            continue  # Skip tokens with both buy and sell counts of zero

        total_buy_counts += counts['buys']
        sol_spent_usd = counts['sol_spent'] * sol_cur_price["price_per_token"]
        sol_made_usd = counts['sol_made'] * sol_cur_price["price_per_token"]
        pnl_usd = sol_made_usd - sol_spent_usd
        pnl_roi = (pnl_usd / sol_spent_usd) * 100 if sol_spent_usd else 0
        
        hold_duration = "N/A"
        held_duration_sec = "N/A"

        if counts['sells'] > counts['buys']:
            tokens_with_more_sells_than_buys += 1
        if counts['first_buy_timestamp'] and counts['last_sell_timestamp']:
            hold_duration_seconds = counts['last_sell_timestamp'] - counts['first_buy_timestamp']

            if hold_duration_seconds > 0:
                total_held_duration_seconds += hold_duration_seconds
                held_duration_sec = hold_duration_seconds
                
                # Calculate the formatted duration
                days = hold_duration_seconds // (24 * 3600)
                hours = (hold_duration_seconds % (24 * 3600)) // 3600
                minutes = (hold_duration_seconds % 3600) // 60
                seconds = hold_duration_seconds % 60
                
                # Create parts array and only add non-zero components
                time_parts = []
                if days > 0:
                    time_parts.append(f"{days} day{'s' if days != 1 else ''}")
                if hours > 0:
                    time_parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
                if minutes > 0:
                    time_parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
                if seconds > 0:
                    time_parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
                
                # Handle the case when all values are zero
                if not time_parts:
                    hold_duration = "0 seconds"
                else:
                    hold_duration = ", ".join(time_parts)
                
                if hold_duration_seconds < 60:
                    tokens_with_short_hold_duration += 1
            elif hold_duration_seconds < 0:
                hold_duration_seconds = 0
                hold_duration = "0 seconds"
                held_duration_sec = 0

            # Check PNL and update streaks
            if pnl_usd > 0:  # Winning trade
                current_winning_streak += 1
                longest_winning_streak = max(longest_winning_streak, current_winning_streak)
                # Reset losing streak
                current_losing_streak = 0  
            elif pnl_usd < 0:  # Losing trade
                current_losing_streak += 1
                longest_losing_streak = max(longest_losing_streak, current_losing_streak)
                # Reset winning streak
                current_winning_streak = 0  
            else:
                # Reset both streaks if it's a break-even trade
                longest_winning_streak = max(longest_winning_streak, current_winning_streak)
                longest_losing_streak = max(longest_losing_streak, current_losing_streak)
                # Reset both current streaks
                current_winning_streak = 0
                current_losing_streak = 0
        else:
            held_duration_sec = "N/A"
            hold_duration = "N/A"

        # When processing token_trade_summary
        token_symbol = token_bought if token_bought_address == token_address else token_sold

        # Or create a dictionary mapping addresses to symbols when processing transactions
        token_address_to_symbol = {}
        for transaction in relevant_transactions:
            token_address_to_symbol[transaction["token_sold_address"]] = transaction["token_sold"]
            token_address_to_symbol[transaction["token_bought_address"]] = transaction["token_bought"]

        # Then use it later when needed
        token_symbol = token_address_to_symbol.get(token_address, "N/A")

        if pnl_roi > 0:
            profitable_trades += 1
            if pnl_roi > best_trade_roi:

                best_trade_roi = pnl_roi
                best_trade = {
                    'token_address': token_address,
                    'symbol': token_symbol,
                    'buys': counts['buys'],
                    'sells': counts['sells'],
                    'buy_amount': counts['total_buy_amount'],
                    'sell_amount': counts['total_sell_amount'],
                    'sol_spent': counts['sol_spent'],
                    'sol_made': counts['sol_made'],
                    'first_buy_timestamp': counts['first_buy_timestamp'],
                    'last_sell_timestamp': counts['last_sell_timestamp'],
                    'hold_duration': hold_duration,
                    'held_duration_sec': held_duration_sec,  # Ensure this is included
                    'profit_loss': counts['sol_made'] - counts['sol_spent'],  # Calculate profit/loss here
                    'profit_loss_usd': pnl_usd,  # Calculate profit/loss here
                    'pnl_roi': pnl_roi,
                }
        else:
            unprofitable_trades += 1
                        
        token_trade_list.append((token_address, counts))
    token_trade_list.sort(key=lambda x: (x[1]['first_buy_timestamp'] is None, x[1]['first_buy_timestamp'] or float('inf')))
    # print("Here 6")

    headers_2 = [
    "Token Address", "Buys", "Sells", "Total Buy Amount", 
    "Total Sell Amount", "PNL (USD)", "PNL ROI (%)", "Held Duration", "First Buy MC (USD)", 
    "Last Sell MC (USD)", "Current MC (USD)", "Scam Score"
    ]

    # Apply bold and center alignment for headers
    for col_num, header in enumerate(headers_2, 1):
        cell = ws.cell(row=current_row, column=col_num, value=header)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    # Start filling data from the second row
    current_row += 1
    for token_address, counts in token_trade_list:
        token_data = results["results"][token_address]
        scam_score = token_data.get("score_normalised", 0)
        

        if scam_score != "N/A":
            scam_score = safe_int_conversion(scam_score)
            if scam_score > 40: 
                tokens_with_high_scam_score += 1
        else:
            scam_score = str("N/A")

        # When processing token_trade_summary
        token_symbol = token_bought if token_bought_address == token_address else token_sold

        # Or create a dictionary mapping addresses to symbols when processing transactions
        token_address_to_symbol = {}
        for transaction in relevant_transactions:
            token_address_to_symbol[transaction["token_sold_address"]] = transaction["token_sold"]
            token_address_to_symbol[transaction["token_bought_address"]] = transaction["token_bought"]

        # Then use it later when needed
        token_symbol = token_address_to_symbol.get(token_address, "N/A")

        total_buy_counts += counts['buys']
        sol_spent_usd = counts['sol_spent'] * sol_cur_price["price_per_token"]
        sol_made_usd = counts['sol_made'] * sol_cur_price["price_per_token"]
        pnl_usd = sol_made_usd - sol_spent_usd
        pnl_roi = (pnl_usd / sol_spent_usd) * 100 if sol_spent_usd else 0
        
        if counts['first_buy_timestamp'] and counts['last_sell_timestamp']:
            hold_duration_seconds = counts['last_sell_timestamp'] - counts['first_buy_timestamp']
            if hold_duration_seconds > 0:
                total_held_duration_seconds += hold_duration_seconds
            elif hold_duration_seconds < 0:
                hold_duration_seconds = 0
                
            if hold_duration_seconds > 0 and hold_duration_seconds < 60:
                tokens_with_short_hold_duration += 1

            if hold_duration_seconds > 0:
                days = hold_duration_seconds // (24 * 3600)
                hours = (hold_duration_seconds % (24 * 3600)) // 3600
                minutes = (hold_duration_seconds % 3600) // 60
                seconds = hold_duration_seconds % 60
                
                # Create parts array and only add non-zero components
                time_parts = []
                if days > 0:
                    time_parts.append(f"{days} day{'s' if days != 1 else ''}")
                if hours > 0:
                    time_parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
                if minutes > 0:
                    time_parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
                if seconds > 0:
                    time_parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
                
                # Handle the case when all values are zero
                if not time_parts:
                    hold_duration = "0 seconds"
                else:
                    hold_duration = ", ".join(time_parts)

            # days = hold_duration_seconds // (24 * 3600)
            # hours = (hold_duration_seconds % (24 * 3600)) // 3600
            # minutes = (hold_duration_seconds % 3600) // 60
            # seconds = hold_duration_seconds % 60
            # held_duration_sec = hold_duration_seconds
            # hold_duration = f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"   
            
            # Check PNL and update streaks
            if pnl_usd > 0:  # Winning trade
                current_winning_streak += 1
                longest_winning_streak = max(longest_winning_streak, current_winning_streak)
                # Reset losing streak
                current_losing_streak = 0  
            elif pnl_usd < 0:  # Losing trade
                current_losing_streak += 1
                longest_losing_streak = max(longest_losing_streak, current_losing_streak)
                # Reset winning streak
                current_winning_streak = 0  
            else:
                # Reset both streaks if it's a break-even trade
                longest_winning_streak = max(longest_winning_streak, current_winning_streak)
                longest_losing_streak = max(longest_losing_streak, current_losing_streak)
                # Reset both current streaks
                current_winning_streak = 0
                current_losing_streak = 0
        else:
            held_duration_sec = "N/A"
            hold_duration = "N/A"
        
        mc_value = counts['first_buy_mc']

        if mc_value is not None:
            try:
                mc_value = float(mc_value)  # Ensure it's a number (float or int)
                if mc_value < 5000:
                    mc_categories["<5K"] += 1
                elif 5000 <= mc_value < 30000:
                    mc_categories["5K-30K"] += 1
                elif 30000 <= mc_value < 100000:
                    mc_categories["30K-100K"] += 1
                elif 100000 <= mc_value < 300000:
                    mc_categories["100K-300K"] += 1
                else:
                    mc_categories["300K+"] += 1
            except ValueError:
                # Handle case where mc_value is not convertible to a number
                print(f"Warning: Market Cap value '{mc_value}' is not a valid number, skipping this token.")
        else:
            # Handle case where mc_value is None
            print(f"warning: Market Cap value is None, skipping this token {token_address}.")    
    

        # Ensure that the values are not None, and replace None with a default value (e.g., 0.0)
        first_buy_mc = counts.get('first_buy_mc', 0.0) if counts.get('first_buy_mc') is not None else 0.0
        last_sell_mc = counts.get('last_sell_mc', 0.0) if counts.get('last_sell_mc') is not None else 0.0
        current_mc = counts.get('current_mc', 0.0) if counts.get('current_mc') is not None else 0.0

        # Format the values to 2 decimal places, ensuring no scientific notation
        formatted_first_buy_mc = f" {first_buy_mc:.2f} "
        formatted_last_sell_mc = f" {last_sell_mc:.2f} "
        formatted_current_mc = f" {current_mc:.2f} "

        ws.append([
            f"{token_address} ({token_symbol})",
            counts['buys'],
            counts['sells'],
            counts['total_buy_amount'],
            counts['total_sell_amount'],
            f"{pnl_usd:.2f}",
            f"{pnl_roi:.2f}",
            hold_duration,
            formatted_first_buy_mc,
            formatted_last_sell_mc,
            formatted_current_mc,
            scam_score
        ])

        for col in range(1, len(headers_2) + 1):
            cell = ws.cell(row=current_row, column=col)
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Apply specific coloring for PNL (USD)
        if pnl_usd > 0:
            ws.cell(row=current_row, column=6).fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")  # Green
        elif pnl_usd < 0:
            ws.cell(row=current_row, column=6).fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")  # Light Red
        elif pnl_usd == 0:
            ws.cell(row=current_row, column=6).fill = PatternFill(start_color="CCFFCC", end_color="CCFFCC", fill_type="solid")  # Light Green

        # Apply specific coloring for PNL (ROI)
        if pnl_roi > 0:
            ws.cell(row=current_row, column=7).fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")  # Green
        elif pnl_roi < 0:
            ws.cell(row=current_row, column=7).fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")  # Light Red
        elif pnl_roi == 0:
            ws.cell(row=current_row, column=7).fill = PatternFill(start_color="CCFFCC", end_color="CCFFCC", fill_type="solid")  # Light Green

        # Highlight Scam Score
        if scam_score != "N/A":
            scam_score = int(scam_score) if str(scam_score).isdigit() else 0  # Ensure it's an integer
            if scam_score > 40:
                ws.cell(row=current_row, column=12).fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")  # Red
            elif scam_score == 40:
                ws.cell(row=current_row, column=12).fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")  # Yellow
            elif scam_score == 0:
                ws.cell(row=current_row, column=12).fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")  # Green
            elif scam_score < 40:
                ws.cell(row=current_row, column=12).fill = PatternFill(start_color="CCFFCC", end_color="CCFFCC", fill_type="solid")  # Light Green
        else:
            # If scam score is "N/A", handle accordingly, e.g., apply a neutral color or leave it empty
            ws.cell(row=current_row, column=12).value = "N/A"
            ws.cell(row=current_row, column=12).alignment = Alignment(horizontal="center", vertical="center")

        # Apply coloring for counts['sells']
        if counts['sells'] > 15:
            ws.cell(row=current_row, column=3).fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")  # Red
        elif counts['sells'] > 4:
            ws.cell(row=current_row, column=3).fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")  # Light Red
        elif counts['sells'] < 4:
            ws.cell(row=current_row, column=3).fill = PatternFill(start_color="CCFFCC", end_color="CCFFCC", fill_type="solid")  # Light Green
        elif counts['sells'] == 0:
            ws.cell(row=current_row, column=3).fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")  # Green

        current_row += 1
    print("here 8")

    def format_duration(seconds):
        """Format seconds into a human-readable duration string showing only non-zero parts."""
        if seconds == 0:
            return "0 seconds"
            
        days = int(seconds // (24 * 3600))
        hours = int((seconds % (24 * 3600)) // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds > 0:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
        
        return ", ".join(parts)

    average_held_duration_seconds = total_held_duration_seconds / total_buy_counts if total_buy_counts else 0
    avg_hld_tme = format_duration(average_held_duration_seconds)

    number_of_tokens_traded = profitable_trades + unprofitable_trades
    average_buy_size = total_sol_spent / number_of_tokens_traded
    average_trades_per_day = number_of_tokens_traded / analyses_period if analyses_period else 0

    winrate = (profitable_trades / number_of_tokens_traded) * 100 if number_of_tokens_traded else 0
    average_buy_per_trade = total_buy_counts / number_of_tokens_traded if number_of_tokens_traded else 0
    
    # Final check after processing all trades
    longest_winning_streak = max(longest_winning_streak, current_winning_streak)
    longest_losing_streak = max(longest_losing_streak, current_losing_streak)
    
    print("here 9")
    # Now add the Best Trade section
    if best_trade:
        ws.append([])
        # Move to the next row
        current_row += 1
        # Add title and header rows
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=14)
        ws.cell(row=current_row, column=1, value="Best Trade Based on ROI").font = Font(size=10, bold=True)
        ws.cell(row=current_row, column=1).alignment = Alignment(horizontal="center", vertical="center")
        # Move to the next row
        current_row += 1

        # Best Trade Information
        headers_3 = [
        "Token Address", "Token Symbol", "Buys/Sells", "Total Buy Amount", 
        "Total Sell Amount", "SOL Spent", "SOL made", "Held Duration", "Profit made (SOL)", "Profit made (USD)", "Profit ROI" 
        ]

        # Apply bold and center alignment for headers
        for col_num, header in enumerate(headers_3, 1):
            cell = ws.cell(row=current_row, column=col_num, value=header)
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")

        current_row += 1
        ws.append([
            best_trade['token_address'],
            best_trade['symbol'],
            f"{best_trade['buys']}/{best_trade['sells']}",
            f"{best_trade['buy_amount']:.2f} {best_trade['symbol']}",
            f"{best_trade['sell_amount']:.2f} {best_trade['symbol']}",
            f"{best_trade['sol_spent']:.9f} SOL",
            f"{best_trade['sol_made']:.9f} SOL",
            best_trade['hold_duration'],
            f"{best_trade['profit_loss']:.9f} SOL",
            f"${best_trade['profit_loss_usd']:.2f}",
            f"{best_trade['pnl_roi']:.2f}%" 
        ])
        
        for col in range(1, len(headers_3) + 1):
            cell = ws.cell(row=current_row, column=col)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")

        current_row += 1
    
    print("here 10")
    ws.append([])
    # Move to the next row
    current_row += 1
    # Add title and header rows
    ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=14)
    ws.cell(row=current_row, column=1, value="Overall Wallet Summary").font = Font(size=10, bold=True)
    ws.cell(row=current_row, column=1).alignment = Alignment(horizontal="center", vertical="center")
    # Move to the next row
    current_row += 1
    
    # Center the text in the cells
    if sol_balance is not None:
        sol_balance = get_sol_balance(address, proxy)  # sol_balance is a dictionary
        sol_balance_value = sol_balance.get("balance", 0.0)  # Extract the float value  
        ws.append(["SOL Balance", f"{sol_balance_value:.9f} SOL"])  # Now it's correctly formatted
        ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center", vertical="center")

    ws.append([])
    current_row += 4
    ws.append(["Number of Tokens Traded", number_of_tokens_traded])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center", vertical="center")
    ws.append(["Profitable Trades", profitable_trades])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center", vertical="center")
    ws.append(["Unprofitable Trades", unprofitable_trades])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center", vertical="center")
    ws.append(["Winrate", f"{winrate:.2f}%"])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center", vertical="center")
    ws.append(["Average Buy Size", f"{average_buy_size:.2f} SOL"])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center", vertical="center")

    ws.append([])
    current_row += 1

    ws.append(["Average Held Duration", avg_hld_tme])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    ws.append(["Average Buys per Trade", f"{average_buy_per_trade:.2f}"])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    ws.append(["Average Trades per Day", f"{average_trades_per_day:.2f}"])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    ws.append(["Last Trade Time", first_transaction_trade_time])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")

    ws.append([])
    current_row += 5
    ws.append(["Average Buy Size (SOL)", f"{average_buy_size:.2f}"])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    ws.append(["Total Gas Fees", f"{total_gas_fees:.9f} SOL"])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    ws.append(["Average Gas Fee per Transaction", f"{average_gas_fee:.9f} SOL"])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")

    ws.append(["Total Jito Fees", f"{total_jito_fees:.9f} SOL"])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    ws.append(["Average Jito Fee per Transaction", f"{average_jito_fee:.9f} SOL"])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")

    profit_loss = total_sol_made - total_sol_spent

    # Create a summary section
    ws.append([])
    current_row += 5
    ws.append(["Total SOL Spent", total_sol_spent])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    ws.append(["Total SOL Made", total_sol_made])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    # ws.append(["Total Gas Fees (SOL)", total_gas_fees])
    # ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    # ws.append(["Total Jito Fees (SOL)", total_jito_fees])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    ws.append(["Transaction Count", transaction_count])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")

    ws.append([])
    current_row += 3
    ws.append(["Tokens with More Sells than Buys", tokens_with_more_sells_than_buys])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    ws.append(["Tokens with Short Hold Duration (< 1 min)", tokens_with_short_hold_duration])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    ws.append(["Scam Tokens", tokens_with_high_scam_score])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")

    # Streak results
    ws.append([])
    current_row += 2
    ws.append(["Longest Winning Streak", longest_winning_streak])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
    ws.append(["Longest Losing Streak", longest_losing_streak])
    ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")

    ws.append([])
    # Move to the next row
    current_row += 12
    # Add title and header rows
    ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=14)
    ws.cell(row=current_row, column=1, value="Market Capitalization Distribution of Tokens").font = Font(size=10, bold=True)
    ws.cell(row=current_row, column=1).alignment = Alignment(horizontal="center", vertical="center")
    # Move to the next row
    current_row += 1

    for category, count in mc_categories.items():
        ws.append([category, f"{count} tokens"])
        ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")

    # Apply bold formatting to the header row
    # for cell in ws[ws.max_row-1]:
    #     cell.font = Font(bold=True)
    
    # Assuming you have the relevant variables defined
    wallet_summary = generate_wallet_summary(
        sol_balance_value, number_of_tokens_traded, profitable_trades,
        unprofitable_trades, winrate, avg_hld_tme, average_buy_per_trade, average_trades_per_day,
        first_transaction_trade_time, average_buy_size, total_sol_spent_usd, total_sol_spent, 
        total_sol_made_usd, total_sol_made, profit_loss, profit_loss_usd, profit_loss_roi, tokens_with_more_sells_than_buys, 
        tokens_with_short_hold_duration, tokens_with_high_scam_score, longest_winning_streak, longest_losing_streak, total_gas_fees, average_gas_fee,
        total_jito_fees, average_jito_fee, address, analyses_period, mc_categories
    )

    best_trade_summary = generate_best_trade(best_trade)
    print("here 10")
    # Send the summaries to the user
    full_message = f"{wallet_summary}\n\n{best_trade_summary}"

    # Apply bold and center alignment for summary
    for cell in ws["E"]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    # Calculate the maximum length of the content in each column
    max_lengths = {col: 0 for col in range(1, len(headers) + 1)}

    # Loop through the rows and find the maximum length of each column
    for row in ws.iter_rows(min_row=2, max_row=current_row - 1, min_col=1, max_col=len(headers)):
        for col_num, cell in enumerate(row, start=1):
            try:
                # Update the max length for the current column
                if cell.value:
                    max_lengths[col_num] = max(max_lengths[col_num], len(str(cell.value)))
            except:
                continue

    # Set the column widths based on the max lengths
    for col_num, max_len in max_lengths.items():
        # Add some padding for better readability (adjust the multiplier as needed)
        adjusted_width = max_len + 2
        col_letter = openpyxl.utils.get_column_letter(col_num)
        ws.column_dimensions[col_letter].width = adjusted_width

    # Save to a file
    file_path = f"full_trade_analysis_{address}.xlsx"
    wb.save(file_path)
    
    # Send the full message and attach the Excel file in one reply
    with open(file_path, "rb") as f:
        await update.message.reply_text(full_message)  # First, send the text message
        await update.message.reply_document(document=f, filename=f"full_trade_analysis_{address}.xlsx")  # Then, send the document

    print(f"Transaction analysis sent to the bot for address: {address}")

# Function to generate top trader
# Function to generate top trader
# Function to generate top trader

async def generate_top_trader_file(update, r, mint_address, relevant_transactions, proxy):
    sol_cur_price = await get_asset(SOL_ADDR, proxy)

    # Track traders' stats
    trader_stats = defaultdict(lambda: {
        "buy_count": 0,
        "sell_count": 0,
        "sol_buy_amount": 0.0,
        "sol_sell_amount": 0.0,
        "pnl_sol": 0.0,
        "pnl_usd": 0.0,
        "roi_percent": 0.0
    })

    # Token metadata processing
    token_addresses = set()


    transactions_list = relevant_transactions
    
    for transaction in transactions_list:
        token_sold = transaction.get("token_sold")
        token_bought = transaction.get("token_bought")

        token_addresses.add(token_sold)
        token_addresses.add(token_bought)
    
    # Initialize Excel workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"Trade - {mint_address[:10]}"

    # Title and headers for transactions section
    ws.merge_cells('A1:N1')
    ws['A1'] = f"Trade Analysis of Wallet {mint_address}"
    ws['A1'].font = Font(size=16, bold=True)
    ws['A1'].alignment = Alignment(horizontal="center", vertical="center")

    # Add transaction headers
    headers = [
        "Signature", "Transaction by", "Token Bought", "Token Bought Amount", 
        "Token Sold", "Token Sold Amount", "Trade Time", "USD WORTH (USD)"
    ]
    ws.append(headers)
    header_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    for cell in ws[2]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    # Populate transaction data and update trader stats
    current_row = 3
    
    # transactions_list = relevant_transactions.get("transactions", [])
    for transaction in transactions_list:
        trader = transaction.get("transaction_by")
        token_sold = transaction.get("token_sold")
        mint_symbol = transaction.get("mint_symbol")
        token_bought = transaction.get("token_bought")
        
        txn_usd_value_chr = transaction.get("txn_usd_value", "$0")
        usd_value_str = float(txn_usd_value_chr.replace("$", "").replace(",", "")) if txn_usd_value_chr and txn_usd_value_chr.lower() != "n/a" else 0.0

        # txn_usd_value = float(transaction.get('txn_usd_value', 0) or 0)
        def safe_float(value):
            try:
                # Handle if value is already a number or a string number.
                return float(value)
            except (ValueError, TypeError):
                return 0.0

        # Then use:
        token_sold_amount = safe_float(transaction.get('token_sold_amount', 0))
        token_bought_amount = safe_float(transaction.get('token_bought_amount', 0))
        # usd_value = safe_float(transaction.get('txn_usd_value', 0))
        usd_value = safe_float(usd_value_str)

        print(f"Updating Trader Stats: {transaction}")
        # Update trader stats
        if token_sold in ["WSOL", "SOL"]:
            trader_stats[trader]["buy_count"] += 1
            trader_stats[trader]["sol_buy_amount"] += token_sold_amount
        elif token_bought in ["WSOL", "SOL"]:
            trader_stats[trader]["sell_count"] += 1
            trader_stats[trader]["sol_sell_amount"] += token_bought_amount
            
            print(f"Updating Trader Stats for SOL: {transaction}")

        # if token_sold not in ["WSOL", "SOL"] and token_sold != mint_symbol:
        #     trader_stats[trader]["buy_count"] += 1
        #     sol_equivalent = usd_value / sol_cur_price["price_per_token"]
        # elif token_bought not in ["WSOL", "SOL"] and token_bought != mint_symbol:
        #     trader_stats[trader]["sell_count"] += 1
        #     sol_equivalent = usd_value / sol_cur_price["price_per_token"]
        #     trader_stats[trader]["sol_sell_amount"] += sol_equivalent
        
        if token_sold not in ["WSOL", "SOL"] and token_sold == mint_symbol:
            trader_stats[trader]["buy_count"] += 1
            sol_equivalent = usd_value / sol_cur_price["price_per_token"]
            trader_stats[trader]["sol_buy_amount"] += sol_equivalent
        elif token_bought not in ["WSOL", "SOL"] and token_bought == mint_symbol:
            trader_stats[trader]["sell_count"] += 1
            sol_equivalent = usd_value / sol_cur_price["price_per_token"]
            trader_stats[trader]["sol_sell_amount"] += sol_equivalent
        
            print(f"Updating Trader Stats for TOKEN: {transaction}")

        ws.append([
            transaction['signature'],
            trader,
            f"{token_bought}",
            transaction['token_bought_amount'],
            f"{token_sold}",
            transaction['token_sold_amount'],
            transaction['trade_time'],
            usd_value
        ])
        current_row += 1

    # Sort traders by ROI
    for trader, stats in trader_stats.items():
        # Calculate PNL (SOL)
        stats["pnl_sol"] = stats["sol_sell_amount"] - stats["sol_buy_amount"]

        # Calculate PNL (USD)
        # stats["pnl_usd"] = stats["pnl_sol"] * sol_cur_price
        stats["pnl_usd"] = stats["pnl_sol"] * sol_cur_price["price_per_token"]

        # Calculate ROI (%)
        stats["roi_percent"] = (
            (stats["pnl_sol"] / stats["sol_buy_amount"]) * 100
            if stats["sol_buy_amount"] > 0
            else 0
        )

    sorted_traders = sorted(
        trader_stats.items(),
        key=lambda x: x[1]["pnl_usd"],
        reverse=True
    )

    # Add Top Traders section
    ws.merge_cells(start_row=current_row + 2, start_column=1, end_row=current_row + 2, end_column=6)
    ws.cell(row=current_row + 2, column=1, value="Top Traders").font = Font(size=14, bold=True)
    ws.cell(row=current_row + 2, column=1).alignment = Alignment(horizontal="center", vertical="center")

    trader_headers = ["Top Trader", "Buy Count", "Sell Count", 
                      "Total SOLs Spent", "Total SOLs Made", "PNL (SOL)", "PNL (USD)", "ROI (%)"]
    ws.append(trader_headers)
    header_fill = PatternFill(start_color="CCCCFF", end_color="CCCCFF", fill_type="solid")
    for cell in ws[current_row + 3]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    # Populate Top Traders data
    current_row += 4
    for trader, stats in sorted_traders:
        ws.append([
            trader,
            stats["buy_count"],
            stats["sell_count"],
            round(stats["sol_buy_amount"], 4),
            round(stats["sol_sell_amount"], 4),
            round(stats["pnl_sol"], 4),
            f"${stats['pnl_usd']:.2f}",
            f"{round(stats['roi_percent'], 2)}%"
        ])
        current_row += 1

    # Save Excel file
    file_path = f"full_trade_analysis_{mint_address}.xlsx"
    wb.save(file_path)

    # Send file to the user
    with open(file_path, "rb") as f:
        await update.message.reply_document(document=f, filename=f"full_trade_analysis_{mint_address}.xlsx")

    print(f"Transaction analysis sent to the bot for token: {mint_address}")
     
async def fetch_token_data_from_server(update, token_mint, active_proxies):
    """
    Sends token list and proxies to Flask server for processing.
    """
    checkScamTokensurl = f"{SERVER_URL}/checkScamTokens"  # Replace with your actual server URL

    payload = {
        "tokens": token_mint,
        "proxies": active_proxies
    }

    # Send an immediate acknowledgment to the user
    status_message = await update.message.reply_text(
        f"Checking Rugproof..... This may take several minutes..."
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(checkScamTokensurl, json=payload) as resp:
            if resp.status != 202:
                await status_message.edit_text(
                    "Failed to start transaction processing. Please try again later."
                )
                return
            data = await resp.json()
            task_id = data.get("task_id")
    
    transactions = None
    while True:
         async with aiohttp.ClientSession() as session:
             async with session.get(f"{SERVER_URL}/task_status/{task_id}") as resp:
                 data = await resp.json()
                 if data.get("state") == "SUCCESS":
                    transactions = data.get("result")
                    # Update the status message when done
                    await status_message.edit_text("Processing complete, fetching results...")
                    break
                 elif data.get("state") == "FAILURE":
                    await status_message.edit_text(
                        "Transaction processing failed. Please try again later."
                    )
                    return
         # Wait before polling again (e.g., 5 seconds)
         await sleep(5)
    
    # Once the background task completes, process the results
    if transactions:
       return transactions
    else:
        await update.message.reply_text("No relevant transactions.")
        return

async def get_asset(sol_address, proxy):
    getAsseturl = f"{SERVER_URL}/get_asset"  # Replace with your actual server IP

    payload = {
        "sol_address": sol_address,
        "proxy": proxy
    }

    # Use aiohttp to make an async POST request
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(getAsseturl, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error from server: {response.status}, {await response.text()}")
                    return {"error": f"Server returned status {response.status}"}
        except Exception as e:
            print(f"Error sending request at get_asset: {e}")
            return {"error": "Request failed"}

async def fetch_account_transactions_from_server_wal(update, address, proxies, analyses_period, active_proxies, proxy):
    start_time = time.time()
    fetchAccountTransurl = f"{SERVER_URL}/fetchAccountTrans"

    # Send an immediate acknowledgment to the user
    status_message = await update.message.reply_text(
        f"Fetching transactions for {address}. This may take several minutes..."
    )

    payload = {
        "address": address,
        "proxies": proxies,
        "analyses_period": analyses_period,
        "active_proxies": active_proxies
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(fetchAccountTransurl, json=payload) as resp:
            if resp.status != 202:
                await status_message.edit_text("Failed to start transaction processing. Please try again later.")
                return
            data = await resp.json()
            task_id = data.get("task_id")
        
    transactions = None
    while True:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SERVER_URL}/task_status/{task_id}") as resp:
                data = await resp.json()
                if data.get("state") == "SUCCESS":
                    transactions = data.get("result")
                    # Update status message once done
                    await status_message.edit_text("Transaction processing complete. Processing results...")
                    break
                elif data.get("state") == "FAILURE":
                    await status_message.edit_text("Transaction processing failed. Please try again later.")
                    return
        # Wait before polling again (e.g., 5 seconds)
        await sleep(5)

    # Once the background task completes, process the results
    if transactions:
        unique_token_addresses = extract_unique_tokens(transactions)
        # Use Flask server to process token data with multiple proxies
        print(f"Fetching metadata for tokens: {unique_token_addresses}")
        unique_token_addres = list(set(unique_token_addresses))  # Remove duplicates
        print(f"Unique token addresses counts: {len(unique_token_addres)}")
        await status_message.edit_text(f"Transaction processing complete. Processing results for {len(unique_token_addres)} tokens traded by {address}...")

        results = await fetch_token_data_from_server(update, unique_token_addresses, active_proxies)

        if "error" in results:
            await update.message.reply_text(f"Error fetching metadata: {results['error']}")
            return

        # await fetch_the_metadata(unique_token_addresses, proxies)

        # Save and process account transactions with metadata results
        await save_account_transactions_to_file(update, address, transactions, analyses_period, proxy, results)
    else:
        await update.message.reply_text(f"No relevant transactions found for {address}.")
        return

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"fetch_and_save_transactions took {elapsed_time:.2f} seconds to complete")

async def fetch_the_metadata(unique_token_addresses, proxy=None):
    """
    Sends token list and proxies to Flask server for processing.
    """
    fetchMetadataurl = f"{SERVER_URL}/fetchMetadata"  # Replace with your actual server IP

    payload = {
        "unique_token_addresses": list(unique_token_addresses)
    }

    # Use aiohttp to make an async POST request
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(fetchMetadataurl, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error from server: {response.status}, {await response.text()}")
                    return {"error": f"Server returned status {response.status}"}
        except Exception as e:
            print(f"Error sending request at Fetch_the_metadata: {e}")
            return {"error": "Request failed"}

# for main file-top_traders
# for main file-top_traders
# for main file-top_traders

# Main function to gather, process, and fetch metadata

async def gather_process_fetch_with_multiple_proxies_top(r, update, mint_address, active_proxies, proxies):
    start_time = time.time()
    fetchTokenDefiTransurl = f"{SERVER_URL}/fetchTokenDEFITrans"
    
    # Send an immediate acknowledgment to the user
    status_message = await update.message.reply_text(
        f"Fetching transactions for {mint_address}. This may take several minutes..."
    )
    
    # Call the Flask endpoint to enqueue the task
    payload = {
        "active_proxies": active_proxies,
        "proxies": proxies,
        "mint_address": mint_address
    }
    
    async with aiohttp.ClientSession() as session:
         async with session.post(fetchTokenDefiTransurl, json=payload) as resp:
             if resp.status != 202:
                 await status_message.edit_text("Failed to start transaction processing. Please try again later.")
                 return
             data = await resp.json()
             task_id = data.get("task_id")
    
    # Poll the task status until the background task completes
    transactions = None
    while True:
         async with aiohttp.ClientSession() as session:
             async with session.get(f"{SERVER_URL}/task_status/{task_id}") as resp:
                 data = await resp.json()
                 if data.get("state") == "SUCCESS":
                     transactions = data.get("result")
                     break
                 elif data.get("state") == "FAILURE":
                     await status_message.edit_text("Transaction processing failed. Please try again later.")
                     return
         # Wait before polling again (e.g., 5 seconds)
         await sleep(5)
    
    # Once the background task completes, process the results
    if transactions:
        with open(f"{mint_address}_address.json", "w") as f:
            json.dump(transactions, f, indent=2)
        #  unique_token_addresses = extract_unique_tokens(transactions)
        await generate_top_trader_file(update, r, mint_address, transactions, proxies)
        await update.message.reply_text(f"Top traders data for token {mint_address} has been successfully fetched.")
    else:
         await update.message.reply_text(f"No relevant transactions found for {mint_address}.")
         return

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"fetch_and_save_transactions took {elapsed_time:.2f} seconds to complete")

async def fetch_active_proxies(proxies):
    """
    Calls the Flask API to filter active proxies.
    """
    response = requests.post(f"{SERVER_URL}/checkProxies", json={"proxies": proxies})
    
    if response.status_code == 200:
        return response.json().get("active_proxies", [])
    else:
        print(f"Error fetching active proxies: {response.text}")
        return []

async def fetch_and_save_for_multiple_addresses_wal(update, addresses, analyses_period, proxies):
    """
    Fetch active proxies from the server and distribute tasks across them.
    """
    active_proxies = await fetch_active_proxies(proxies)  # Get valid proxies from Flask

    if not active_proxies:
        await update.message.reply_text("No active proxies available.")
        return

    random.shuffle(active_proxies)
    proxy_cycle = cycle(active_proxies)  # Create a cycle of proxies

    tasks = []
    for address in addresses:
        proxy = next(proxy_cycle)  # Get the next available proxy

        if proxy not in proxy_usage:  # Ensure the proxy has a semaphore
            proxy_usage[proxy] = Semaphore(1)

        async with proxy_usage[proxy]:  # Use the semaphore to avoid overload
            tasks.append(
                fetch_account_transactions_from_server_wal(update, address, proxies, analyses_period, active_proxies, proxy)
            )

    await gather(*tasks)  # Execute all tasks concurrently

# for main file-top_traders
# for main file-top_traders
# for main file-top_traders
async def fetch_and_save_for_multiple_mint_addresses_top(r, update, mint_addresses, proxies):
    """
    Fetch active proxies from the server and distribute tasks across them.
    """

    # Ensure mint_addresses is a list
    if isinstance(mint_addresses, str):
        mint_addresses = [mint_addresses]  # Wrap single address in a list

    # Get active proxies from Flask server
    active_proxies = await fetch_active_proxies(proxies)

    if not active_proxies:
        await update.message.reply_text("No active proxies available.")
        return

    random.shuffle(active_proxies)
    proxy_cycle = cycle(active_proxies)  # Create a cycle of active proxies

    tasks = []
    for mint_address in mint_addresses:  # Iterate over the list of addresses
        # Get the next available proxy that isn't overused
        proxy = await get_next_available_proxy(proxy_cycle)

        # Ensure proxy has a semaphore before using it
        if proxy not in proxy_usage:
            proxy_usage[proxy] = Semaphore(1)

        # Limit usage of the proxy using semaphore
        print("here")
        async with proxy_usage[proxy]:
            tasks.append(
                gather_process_fetch_with_multiple_proxies_top(
                    r, update, mint_address, active_proxies, proxy
                )
            )

    # Gather all tasks and await their completion
    await gather(*tasks)

async def getDaysLeft(user_id):
    """Fetch the remaining subscription days for a user."""
    try:
        url = f"https://bagscan-bot.vercel.app/api/days_left/{user_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logging.info(f"Days left for user {user_id}: {data}")
                    return data  # Return the entire response JSON
                else:
                    logging.warning(f"Failed to fetch days left for user {user_id}: {response.status}")
                    return {"message": "Failed to fetch days left", "data": {}}
    except Exception as e:
        logging.error(f"Error fetching days left for user {user_id}: {e}")
        return {"message": "Error occurred", "data": {}}


async def getUserPlan(user_id):
    url = f"https://bagscan-bot.vercel.app/api/user/{user_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                logging.info(f"Fetching user data for ID {user_id}, status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    logging.info(f"User data fetched successfully: {data}")
                    return data
                else:
                    logging.warning(f"Failed to fetch user data. Status: {response.status}")
                    return None
    except Exception as e:
        logging.error(f"Error in getUserPlan: {e}")
        return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user

    data = {
        "firstName": user.first_name,
        "lastName": user.last_name,
        "username": user.username,
        "id": user.id,
        "is_bot": user.is_bot,
    }

    user_id = user.id
    await createNewUser(data)

    # Fetch user data and plan
    user_data = await getUserPlan(user_id)
    plan = "Free"  # Default plan
    status = "Inactive"  # Default status
    if user_data and user_data.get("data"):
        subscription = user_data["data"].get("subscription", {})
        plan_details = subscription.get("plan", {})
        plan = plan_details.get("plan", "Free").capitalize()
        status = subscription.get("status", "Inactive").capitalize()

    # Get user details
    username = user.first_name or user.username or "there"

    # Fetch subscription status and days left
    days_left_response = await getDaysLeft(user.id)
    days_left_message = days_left_response.get("message", "N/A")
    days_left = days_left_response.get("data", {}).get("daysLeft", "N/A")
    formatted_date = days_left_response.get("data", {}).get("formattedDate", "N/A")

    # Fetch subscription status
    subscription_status = await checkSubscription(user.id)

    # Construct subscription information
    if subscription_status.get("hasSubscription"):
        subscription_info = (
            f"🎉 *Subscription Active!*\n"
            f"- Plan: *{plan}*\n"
            f"- Status: *{status}*\n"
            f"- Days left: *{days_left}*\n"
            f"- Expires on: *{formatted_date}*\n\n"
        )
    else:
        subscription_info = (
            f"🚀 *No active subscription.*\n"
            "👉 Tap '🔒 Subscribe' below to unlock premium features.\n\n"
        )

    # Greeting message with subscription status
    greeting_message = (
        f"🤖 Hey {username}, welcome to *BagScan* - your ultimate wallet and token analytics bot! 🚀\n\n"
        f"{subscription_info}"
        "🌟 *What you can do here:*\n"
        "✅ *Analyze Wallets*: Track trading stats and trends effortlessly.\n"
        "✅ *Discover Top Holders*: Fetch detailed token holder data with ease.\n"
        "✅ *Explore Top Traders*: Identify influential traders and their activity.\n"
        "✅ *Real-Time Insights*: Stay ahead with live wallet and token analysis.\n\n"
        "👇 Select an option below to get started:"
    )

    # Define the welcome keyboard with the Subscribe button
    keyboard = [
        [
            InlineKeyboardButton("📊 Analyze Wallets", callback_data="analyze_wallets"),
            InlineKeyboardButton("💰 Fetch Top Holders", callback_data="fetch_holders"),
        ],
        [
            InlineKeyboardButton("📈 Fetch Top Traders", callback_data="fetch_top_holders"),
            InlineKeyboardButton("💡 Help & Instructions", callback_data="help_instructions"),
        ],
        [
            InlineKeyboardButton("🔗 Referrals", callback_data="referrals"),
            InlineKeyboardButton("🛠 Backup Bots", callback_data="backup_bots"),
        ],
        [
            InlineKeyboardButton("🔒 Subscribe", callback_data="subscribe"),
            InlineKeyboardButton("⚙️ Settings", callback_data="open_settings"),
        ],
        [
            InlineKeyboardButton("❌ Dismiss", callback_data="dismiss_message"),
        ],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    # Send the welcome message
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=greeting_message,
        parse_mode="Markdown",
        reply_markup=reply_markup,
    )



from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ForceReply

async def subscription_modal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = query.message.chat.id

    admin_usernames = ["admin1", "admin2"]  # Replace with actual admin usernames
    admin_contacts = ", ".join([f"@{username}" for username in admin_usernames])

    subscription_message = (
        "✨ *Subscription Plans:*\n\n"
        "🎁 *Free Tier:*\n- Limited features\n\n"
        "💎 *Monthly Plan:*\n- $10/month\n- Full access\n\n"
        "👑 *Yearly Plan:*\n- $100/year (Save 20%)\n- Full access\n\n"
        f"📩 Contact admins for a coupon code: {admin_contacts}\n\n"
        "👉 Enter your coupon code below to activate your subscription."
    )

    # Define navigation keyboard
    keyboard = [
        [
            InlineKeyboardButton("🔙 Menu", callback_data="open_main_menu"),
            InlineKeyboardButton("❌ Dismiss", callback_data="dismiss_message")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Send the subscription message with the navigation keyboard
    await context.bot.send_message(
        chat_id=chat_id,
        text=subscription_message,
        parse_mode="Markdown",
        reply_markup=reply_markup,  # Now properly defined
    )

    # Ask the user to enter their coupon code
    await context.bot.send_message(
        chat_id=chat_id,
        text="🔑 Please enter your coupon code (e.g., ABC123):",
        reply_markup=ForceReply(),  # Force the user to reply
    )


# function to process the coupon code

async def process_coupon_code(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    chat_id = update.message.chat_id
    coupon_code = update.message.text.strip()  # Get the coupon code

    try:
        # Notify backend about the subscription activation
        url = "https://bagscan-bot.vercel.app/api/subscribe"
        payload = {"userId": user.id, "coupon": coupon_code}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                # Log response details
                logging.info(f"Response status: {response.status}")
                response_text = await response.text()
                logging.info(f"Response text: {response_text}")

                if response.status in [200, 201]:  # Check for both 200 and 201 status codes
                    data = await response.json()
                    if data.get("success"):
                        # Notify user of successful subscription
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=(
                                "✅ *Subscription Activated!*\n"
                                "🎉 You now have full access to BagScan's premium features.\n\n"
                                "Thank you for subscribing! 🚀"
                            ),
                            parse_mode="Markdown",
                        )
                    else:
                        # Handle backend response errors
                        error_message = data.get("message", "Unknown error occurred.")
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=( 
                                f"❌ *Failed to activate subscription:*\n"
                                f"{error_message}\n\n"
                                "Please contact an admin for assistance."
                            ),
                            parse_mode="Markdown",
                        )
                else:
                    # Handle non-200/201 response
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"❌ *Error activating subscription:*\n"
                            f"Server responded with status code {response.status}.\n"
                            "Please try again later or contact support."
                        ),
                    )
    except Exception as e:
        logging.error(f"Error in process_coupon_code: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ An unexpected error occurred. Please try again later.",
        )

def subscription_required(func):
    @wraps(func)
    async def wrapper(update, context, *args, **kwargs):
        user_id = update.message.from_user.id
        subscription_status = await checkSubscription(user_id)

        if not subscription_status.get("hasSubscription"):
            # Inform the user they need to subscribe
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=(
                    "❌ *Access Denied: Subscription Required!*\n"
                    "To access this feature, you need an active subscription. "
                    "Please subscribe to continue. 🚀"
                ),
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔒 Subscribe", callback_data="subscribe")]
                ])
            )
            return  # Prevent the original command from executing

        # Proceed with the original function if the user has a valid subscription
        return await func(update, context, *args, **kwargs)
    return wrapper


async def button_handler(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()  # Acknowledge the callback
    logger.info(f"Callback query received: {query.data}")

    if query.data == "fetch_holders":
        await query.message.reply_text("Please enter the mint address:")
        context.user_data["awaiting_mint"] = "top_holders"
        logger.info("Awaiting mint address for top holders.")
    elif query.data == "fetch_top_holders":
        await query.message.reply_text("Please enter the mint address:")
        context.user_data["awaiting_mint"] = "top_traders"
        logger.info("Awaiting mint address for top traders.")



# Capture the mint address from the user
async def capture_mint_address(update: Update, context: CallbackContext) -> None:
    if context.user_data.get("awaiting_mint"):
        mint_address = update.message.text.strip()
        context.user_data["awaiting_mint"] = False  # Reset the flag
        logger.info(f"Received mint address: {mint_address}")
        await update.message.reply_text(f"Fetching top holders for token: {mint_address}...")
        await top_holders(update, context, mint_address)
    else:
        logger.warning("Received unexpected message without awaiting mint address.")
        await update.message.reply_text("Please use the button to fetch top holders.")

async def get_token_mets(mint):
    url = GET_ASSET_URL
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers={'Content-Type': 'application/json'},
            data=json.dumps({
                'jsonrpc': '2.0',
                'id': 'my-id',
                'method': 'getAsset',
                'params': {
                    'id': mint,
                    'displayOptions': {
                        'showFungible': True
                    }
                }
            })
        ) as response:
            result = await response.json()
            logger.info(f"Response from getAsset: {result}")

            asset = result.get('result', {})
            print("asset")
            print(asset)
            if not asset:
                logger.error("Asset not found in response.")
                return None

            token_info = asset.get('token_info', {})
            print("Price INFO")
            print(token_info.get('price_info', {}))
            price_per_token = token_info.get('price_info', {}).get('price_per_token', None)
            token_supply = token_info.get('supply', None)
            token_decimals = token_info.get('decimals', None)

            if token_supply is None or token_decimals is None:
                logger.error("Missing token info in the response.")
                return None

            # market_cap = price_per_token * token_supply / 10**token_decimals
            # logger.info(f"Calculated market cap: {market_cap}")

            return token_decimals, token_supply


# Function to find top holders
async def find_top_holders(mint):
    token_mets = await get_token_mets(mint)
    if not token_mets:
        logger.error("Failed to get token metadata. Exiting.")
        return None

    met1, met2 = token_mets  # Unpack safely since it's not None

    try:
        url = GET_ASSET_URL
        page = 1
        all_owners = {}
        total_fetched = 0

        while True:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "jsonrpc": "2.0",
                    "method": "getTokenAccounts",
                    "id": "helius-test",
                    "params": {
                        "page": page,
                        "limit": 1000,
                        "displayOptions": {},
                        "mint": mint
                    }
                })
            )

            if response.status_code != 200:
                raise Exception(f"Error fetching data: {response.status_code}, {response.text}")

            data = response.json()
            logger.info(f"Fetched data for page {page}: {data}")

            if not data.get("result") or not data["result"].get("token_accounts"):
                logger.info("No more token accounts found.")
                break

            for account in data["result"]["token_accounts"]:
                owner = account["owner"]
                balance = int(account.get("amount", 0)) / 10 ** met1
                met2_ = met2 / 10**met1
                percent_sup = (balance / met2_) * 100

                percent_sup = round(percent_sup, 2)

                if owner in all_owners:
                    all_owners[owner]["balance"] += balance
                    all_owners[owner]["percent_supply"] += percent_sup
                else:
                    all_owners[owner] = {
                        "balance": balance,
                        "percent_supply": percent_sup
                    }

            total_fetched += len(data["result"]["token_accounts"])
            logger.info(f"Total fetched so far: {total_fetched}")
            page += 1

        sorted_owners = sorted(all_owners.items(), key=lambda x: x[1]["balance"], reverse=True)

        top_holders = [
            {
                "owner": owner,
                "balance": details["balance"],
                "percent_supply": details["percent_supply"]
            }
            for owner, details in sorted_owners
        ]

        with open("top_holders.json", "w") as f:
            json.dump(top_holders, f, indent=2)

        logger.info("Top holders data saved to top_holders.json.")
        return top_holders
    except Exception as e:
        logger.error(f"Error in find_top_holders function: {e}")
        return None

# Function to generate Excel file
def generate_excel(mint, data, met1, met2):
    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "Top Token Holders"

        # Add the main title
        title = f"Top Holders of {mint}"
        ws.merge_cells('A1:D1')  # Merge cells A1 to D1
        title_cell = ws['A1']
        title_cell.value = title

        # Center-align the title
        title_cell.alignment = Alignment(horizontal='center', vertical='center')

        # Add headers to the Excel file
        headers = ["S/N", "Account", "Balance", "Percentage"]
        ws.append(headers)

        # Set the alignment for header row (center alignment)
        for col in range(1, 5):  # Columns A to D
            cell = ws.cell(row=1, column=col)
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Add the data to the Excel sheet
        red_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
        for index, entry in enumerate(data, start=1):
            row = [index, entry['owner'], entry['balance'], entry['percent_supply']]
            ws.append(row)

            # Apply conditional formatting: If balance or percentage > 5, color red
            balance_cell = ws.cell(row=index + 1, column=3)  # Balance is in column C
            percentage_cell = ws.cell(row=index + 1, column=4)  # Percentage is in column D
            met2_ = met2 / 10**met1
            exceed_five_percent = (5 * met2_) / 100  # calculate threshold for 5% of total supply

            if entry['balance'] >= exceed_five_percent:  # Compare balance with the 5% threshold
                balance_cell.fill = red_fill
            if entry['percent_supply'] >= 5:  # Directly check if percent supply exceeds 5%
                percentage_cell.fill = red_fill

            # Center-align all cells
            for col in range(1, 5):
                cell = ws.cell(row=index + 1, column=col)
                cell.alignment = Alignment(horizontal="center", vertical="center")

        # Set column widths to auto-fit based on content length
        for col in range(1, 5):  # Columns A to D
            max_length = 0
            column = chr(64 + col)  # Get the letter for column (e.g., A, B, C, D)
            for cell in ws[column]:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except Exception as e:
                    logger.error(f"Error in calculating column width: {e}")

            adjusted_width = max_length + 2  # Add some padding
            # Apply a cap on width to prevent it from becoming too wide
            max_column_width = 30
            ws.column_dimensions[column].width = min(adjusted_width, max_column_width)

        # Save to an Excel file with absolute path
        file_name = f"top_holders_{mint}.xlsx"
        file_path = path.abspath(file_name)
        wb.save(file_path)
        logger.info(f"Excel file saved at {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error generating Excel file: {e}")
        return None

# Function to handle top_holders command
@subscription_required
async def top_holders(update: Update, context: CallbackContext, mint: str = None) -> None:
    # Get the mint address from context or arguments
    if not mint:
        mint = context.args[0] if context.args else None
    if not mint:
        await update.message.reply_text('Please provide a mint address.')
        logger.warning("No mint address provided.")
        return

    await update.message.reply_text(f"Fetching top holders for token: {mint}...")
    await update.message.reply_text("Fetching data for top holders... Please wait.")
    logger.info(f"Fetching top holders for mint address: {mint}")

    result = await find_top_holders(mint)
    if result is None:
        await update.message.reply_text("Failed to fetch top holders. Please try again later.")
        return

    met_data = await get_token_mets(mint)
    if met_data is None:
        await update.message.reply_text("Failed to fetch token metadata. Please check the mint address.")
        return

    met1, met2 = met_data

    try:
        if result:
            await update.message.reply_text("Generating Excel file...")
            file_path = generate_excel(mint, result, met1, met2)
            if file_path is None:
                await update.message.reply_text("Failed to generate Excel file.")
                return

            # Store file path in user context for later use
            context.user_data["excel_file"] = file_path
            logger.info(f"Excel file stored in user context: {file_path}")

            # Provide a button for downloading
            keyboard = [
                [InlineKeyboardButton("Download Excel File", callback_data="download_excel")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "Top holders data is ready! Click the button below to download the Excel file.",
                reply_markup=reply_markup
            )
            logger.info("Download button sent to user.")
        else:
            await update.message.reply_text("There was an error fetching top holders.")
            logger.error("Result from find_top_holders is empty.")
    except Exception as e:
        logger.error(f"Error in top_holders function: {e}")
        await update.message.reply_text(f"An error occurred: {str(e)}")


@subscription_required
async def top_traders(update: Update, context: CallbackContext, mint: str) -> None:
    if not mint:
        await update.message.reply_text("Please provide a mint address.")
        logger.warning("No mint address provided.")
        return

    await update.message.reply_text(f"Fetching top traders for token: {mint}...")
    logger.info(f"Fetching top traders for mint address: {mint}")

    proxies = await load_proxies('proxies.txt')  # Ensure this returns a list
    r = await create_redis_connection_pool()  # Await Redis connection pool

    try:
        # Fetch data for top traders
        await update.message.reply_text("Fetching data for top traders... Please wait.")

        # Do:
        create_task(fetch_and_save_for_multiple_mint_addresses_top(r, update, mint, proxies))

        # await update.message.reply_text(f"Top traders data for token {mint} has been successfully fetched.")
    except Exception as e:
        logger.error(f"Error in top_traders function: {e}")
        await update.message.reply_text(f"An error occurred: {str(e)}")
    finally:
        await r.close()



async def download_report_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()  # Acknowledge the callback

    if query.data == "download_traders":
        file_path = context.user_data.get("traders_excel_file")
        if file_path:
            with open(file_path, "rb") as file:
                await query.message.reply_document(document=file)
                logger.info("Traders Excel file sent to user.")
        else:
            await query.message.reply_text("No traders report available to download.")
            logger.warning("No traders Excel file found in user context.")
    elif query.data == "download_holders":
        file_path = context.user_data.get("excel_file")
        if file_path:
            with open(file_path, "rb") as file:
                await query.message.reply_document(document=file)
                logger.info("Holders Excel file sent to user.")
        else:
            await query.message.reply_text("No holders report available to download.")
            logger.warning("No holders Excel file found in user context.")





# Handler for downloading the Excel file
async def download_excel(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()  # Acknowledge the callback
    logger.info("Download Excel button clicked.")

    # Get the file path from user context
    file_path = context.user_data.get("excel_file")
    if file_path:
        if path.exists(file_path):
            try:
                with open(file_path, 'rb') as file:
                    await query.message.reply_document(
                        document=file,
                        filename=path.basename(file_path)
                    )
                await query.message.reply_text("File sent successfully!")
                logger.info(f"Excel file sent to user: {file_path}")
            except Exception as e:
                logger.error(f"Error sending file: {e}")
                await query.message.reply_text("Failed to send the file. Please try again.")
        else:
            logger.error(f"File does not exist: {file_path}")
            await query.message.reply_text("File not found. Please generate it again.")
    else:
        logger.warning("No file path found in user context.")
        await query.message.reply_text("No file found to download.")

# Global error handler
async def error_handler(update: object, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(f"An error occurred: {str(context.error)}")


# --- Settings Modal ---
async def open_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display settings and analysis period options."""
    query = update.callback_query

    if query:  # If it's a callback query
        await query.answer()
        chat_id = query.message.chat.id
        message = query.message
    else:  # If it's a command
        chat_id = update.message.chat.id

    # Ensure user_data exists and set the default analysis period to 1 days
    if chat_id not in user_data:
        user_data[chat_id] = {"analyses_period": 1}  # Default to 1 day
    else:
        user_data[chat_id].setdefault("analyses_period", 1)  # Set default if not present

    analyses_period = user_data[chat_id]["analyses_period"]

    # Define keyboard
    keyboard = [
        [InlineKeyboardButton("— Trades Period —", callback_data="no_action")],
        [
            InlineKeyboardButton(
                f"✅ 1 Day" if analyses_period == 1 else "1 Day", callback_data="period_1"
            ),
            InlineKeyboardButton(
                f"✅ 7 Days" if analyses_period == 7 else "7 Days", callback_data="period_7"
            ),
        ],
        [
            InlineKeyboardButton(
                f"✅ 14 Days" if analyses_period == 14 else "14 Days", callback_data="period_14"
            ),
            InlineKeyboardButton(
                f"✅ 30 Days" if analyses_period == 30 else "30 Days", callback_data="period_30"
            ),
        ],
        [
            InlineKeyboardButton("🔙 Menu", callback_data="open_main_menu"),
            InlineKeyboardButton("❌ Close", callback_data="dismiss_message")
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    # Edit or send the settings message
    settings_text = "⚙️ *Bot Settings*\n\nSelect your preferred analysis period:"
    if query:
        await message.edit_text(settings_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    else:
        await context.bot.send_message(chat_id=chat_id, text=settings_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

# --- Welcome Modal Triggered from Menu ---
async def show_welcome_modal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display the welcome modal when 'Menu' is clicked."""
    query = update.callback_query
    await query.answer()

    # Updated keyboard with "🔒 Subscribe" button
    keyboard = [
        [
            InlineKeyboardButton("📊 Analyze Wallets", callback_data="analyze_wallets"),
            InlineKeyboardButton("💰 Fetch Top Holders", callback_data="fetch_holders"),
        ],
        [
            InlineKeyboardButton("📈 Fetch Top Traders", callback_data="fetch_top_holders"),
            InlineKeyboardButton("💡 Help & Instructions", callback_data="help_instructions"),
        ],
        [
            InlineKeyboardButton("🔗 Referrals", callback_data="referrals"),
            InlineKeyboardButton("🛠 Backup Bots", callback_data="backup_bots"),
        ],
        [
            InlineKeyboardButton("🔒 Subscribe", callback_data="subscribe"),  # Added Subscribe button
            InlineKeyboardButton("⚙️ Settings", callback_data="open_settings"),
        ],
        [
            InlineKeyboardButton("❌ Dismiss", callback_data="dismiss_message"),
        ],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.edit_text(
        f"🌟 *What you can do here:*\n"
        "✅ *Analyze Wallets*: Track trading stats and trends effortlessly.\n"
        "✅ *Discover Top Holders*: Fetch detailed token holder data with ease.\n"
        "✅ *Explore Top Traders*: Identify influential traders and their activity.\n"
        "✅ *Real-Time Insights*: Stay ahead with live wallet and token analysis.\n\n"
        "👇 Select an option below to get started:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
    )



async def dismiss_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler to dismiss a message."""
    query = update.callback_query
    await query.answer()
    await query.message.delete() 

# --- Callback Query Handler ---
async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button interactions."""
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat.id

    if query.data == "no_action":
        return  # Do nothing

    if query.data.startswith("period_"):
        # Handle analysis period selection
        selected_period = query.data.split("_")[1]
        user_data.setdefault(chat_id, {})["analyses_period"] = (
            int(selected_period) if selected_period.isdigit() else "max"
        )
        await open_settings(update, context)
    elif query.data == "open_settings":
        await open_settings(update, context)
    elif query.data == "open_main_menu":
        await show_welcome_modal(update, context)
    elif query.data == "dismiss_message":
        await query.message.delete()




# Help & Instructions callback handler
async def help_instructions(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    """Handler for /help command and Help & Instructions button."""
    # Define the keyboard
    keyboard = [
        [
            InlineKeyboardButton("🔙 Menu", callback_data="open_main_menu"),
            InlineKeyboardButton("❌ Dismiss", callback_data="dismiss_message")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Check if this is a callback query or a command
    query = update.callback_query

    # Define the help message text
    help_message = (
        "💡 *Help & Instructions*\n\n"
        "Welcome to the Wallet Analysis Bot! Here's how to make the most of it:\n\n"
        "1️⃣ *Analyze Wallets*: Tap 📊 Analyze Wallets and provide a wallet address to view trading activities.\n"
        "2️⃣ *Fetch Top Holders*: Use 💰 Fetch Top Holders to see wallets with the largest holdings.\n"
        "3️⃣ *Fetch Top Traders*: Use 📈 Fetch Top Traders to identify the most active traders.\n"
        "4️⃣ *Set Periods*: Adjust analysis durations and preferences via ⚙️ Settings.\n"
        "5️⃣ *Track Trends*: Monitor real-time insights for the supported blockchain, *Solana*.\n"
        "6️⃣ *Referrals*: Tap 🔗 Referrals to invite friends and earn rewards.\n"
        "7️⃣ *Backup Bots*: Access 🛠 Backup Bots to view alternative bot options.\n\n"
        "Use the buttons below to return to the menu or dismiss this message."
    )

    if query:
        # Handle the callback query
        await query.answer()
        await query.message.edit_text(
            text=help_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    else:
        # Handle the /help command
        await update.message.reply_text(
            text=help_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup  # Include the keyboard here as well
        )


# Handle callback from the Start button
async def handle_start_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await start(query, context)



async def handle_period_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user selecting an analysis period."""
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat.id
    period = int(query.data.split("_")[1])  # Extract the period from callback_data

    # Update the user's selected analysis period
    user_data[chat_id] = {"analyses_period": period}

    await open_settings(update, context)


async def gridbutton_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard button clicks with subscription checks."""
    query = update.callback_query
    user_id = query.from_user.id  # Get the user ID
    await query.answer()

    if query.data == "open_settings":
        await open_settings(update, context)

    elif query.data == "analyze_wallets":
        # Perform subscription check
        subscription_status = await checkSubscription(user_id)
        if not subscription_status.get("hasSubscription"):
            # User doesn't have an active subscription
            await query.message.reply_text(
                "❌ *Access Denied: Subscription Required!*\n"
                "To use the wallet analysis feature, you need an active subscription. "
                "Please subscribe to continue. 🚀",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔒 Subscribe", callback_data="subscribe")]
                ])
            )
            return  # Exit the function early if the user is not subscribed

        # If subscription is valid, proceed with the action
        await query.message.reply_text("Please enter the wallet address(es) for analysis:")

    elif query.data == "help_instructions":
        await handle_help_instructions(update, context)

    elif query.data == "backup_bots":
        await query.message.reply_text("🛠 *Backup bots will be added soon!*", parse_mode="Markdown")

    elif query.data == "referrals":
        await query.message.reply_text("🔗 Our referral reward system is coming soon!")



async def prompt_continue_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ask the user if they want to continue analysis."""
    keyboard = [
        [
            InlineKeyboardButton("Yes", callback_data="continue_analysis_yes"),
            InlineKeyboardButton("No", callback_data="continue_analysis_no"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.message:
        await update.message.reply_text("Do you want to analyze another wallet?", reply_markup=reply_markup)
    elif update.callback_query:
        await update.callback_query.message.reply_text("Do you want to analyze another wallet?", reply_markup=reply_markup)


async def handle_continue_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user's response to continue analysis prompt."""
    query = update.callback_query
    await query.answer() 

    if query.data == "continue_analysis_yes":
        await query.message.reply_text("Great! Please send the wallet address(es) for analysis:")
    elif query.data == "continue_analysis_no":
        await query.message.reply_text("Thank you for using BagScan! If you need further analysis, just type /start. or use the menu button.")


async def createNewUser(data):
    try:
        response = requests.post("https://bagscan-bot.vercel.app/api/user/register", json=data)
        if response.status_code == 201:
            # Log success for a new user registration
            logging.info(f"✅ User {data['username']} (ID: {data['id']}) registered successfully.")
        elif response.status_code == 409:
            # Log if the user is already registered
            logging.info(f"ℹ️ User {data['username']} (ID: {data['id']}) is already registered.")
        else:
            # Log unexpected responses
            logging.warning(f"⚠️ Unexpected response while registering user {data['username']} (ID: {data['id']}): {response.status_code} - {response.text}")
        return None
    except Exception as e:
        # Log any errors during the process
        logging.error(f"❌ Error while creating new user {data['username']} (ID: {data['id']}): {e}")
        return None


async def checkSubscription(data):
    try:
        # Send a GET request to the subscription status API
        response = requests.get(f"https://bagscan-bot.vercel.app/api/subscription_status/{data}")
        
        if response.status_code == 200:
            # Parse the JSON response
            subscriptionStatus = response.json()
            logging.info(f"Subscription status for user {data}: {subscriptionStatus}")
            
            # Safely return the "data" object from the response or a default structure
            return subscriptionStatus.get("data", {"hasSubscription": False, "daysLeft": None})
        
        # Log non-200 status codes and return a default structure
        logging.warning(f"Non-200 response from subscription API: {response.status_code}")
        return {"hasSubscription": False, "daysLeft": None}
    
    except Exception as e:
        # Log any exceptions that occur and return a default structure
        logging.error(f"Error while checking subscription for user {data}: {e}")
        return {"hasSubscription": False, "daysLeft": None}


async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    chat_id = update.message.chat.id

    # Check if the bot is awaiting a mint address for top holders or top traders
    if context.user_data.get("awaiting_mint"):
        action = context.user_data.pop("awaiting_mint")  # Retrieve and clear the flag
        if action == "top_holders":
            await top_holders(update, context, user_input)  # Pass user_input as mint
        elif action == "top_traders":
            await top_traders(update, context, user_input)  # Pass user_input as mint
        return

    # Ensure user data exists and analysis period is set
    if chat_id not in user_data:
        user_data[chat_id] = {"analyses_period": 1}  # Set default

    analyses_period = user_data[chat_id].get("analyses_period", 30)  # Default to 30 days

    # Split addresses by spaces or commas
    addresses = [address.strip() for address in re.split(r'[,\s]+', user_input) if address]

    if not addresses:
        await update.message.reply_text("No valid wallet addresses found. Please try again.")
        return

    # Remove duplicates by converting to a set and back to a list
    unique_addresses = list(set(addresses))
    if len(unique_addresses) != len(addresses):
        await update.message.reply_text(
            f"Duplicate addresses were removed. Proceeding with the unique addresses: {', '.join(unique_addresses)}"
        )

    proxies = await load_proxies('proxies.txt')  # Ensure this returns a list
    # r = await create_redis_connection_pool()  # Await Redis connection pool

    try:
        logging.info(f"Fetching data for addresses: {unique_addresses} with analysis period: {analyses_period} days")
        await update.message.reply_text(
            f"Fetching data for the following addresses:\n" + "\n".join(unique_addresses) +
            f"\nAnalysis period: {analyses_period} days. Please wait..."
        )
        # await fetch_and_save_for_multiple_addresses_wal(r, update, unique_addresses, analyses_period, proxies)
        await fetch_and_save_for_multiple_addresses_wal(update, unique_addresses, analyses_period, proxies)
        await update.message.reply_text("Data fetching completed successfully!")
        await prompt_continue_analysis(update, context)

    except Exception as e:
        logging.exception("Error while fetching data")
        await update.message.reply_text("An error occurred while fetching the data. Please try again later.")
    # finally:
        # await r.close()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)    
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

   # Command Handlers (specific commands first)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("settings", open_settings))
    application.add_handler(CommandHandler("help", help_instructions))
    application.add_handler(CommandHandler('top_holders', top_holders))
    application.add_handler(CommandHandler('top_traders', top_traders))

# CallbackQuery Handlers (specific patterns first)
    application.add_handler(CallbackQueryHandler(help_instructions, pattern="help_instructions"))
    application.add_handler(CallbackQueryHandler(handle_continue_analysis, pattern="^continue_analysis_"))
    application.add_handler(CallbackQueryHandler(handle_period_selection, pattern="^period_"))
    application.add_handler(CallbackQueryHandler(show_welcome_modal, pattern="open_main_menu"))
    application.add_handler(CallbackQueryHandler(dismiss_message, pattern="dismiss_message"))
    application.add_handler(CallbackQueryHandler(button_handler, pattern="^fetch_holders$"))
    application.add_handler(CallbackQueryHandler(download_excel, pattern="^download_excel$"))
    application.add_handler(CallbackQueryHandler(button_handler, pattern="^fetch_top_holders$"))
    application.add_handler(CallbackQueryHandler(download_report_callback, pattern="^download_traders$"))
    application.add_handler(CallbackQueryHandler(download_report_callback, pattern="^download_holders$"))

      # Add a handler for the subscription modal
    application.add_handler(CallbackQueryHandler(subscription_modal, pattern="^subscribe$"))

    # Add a handler for processing the coupon code
    application.add_handler(MessageHandler(filters.TEXT & filters.REPLY, process_coupon_code))

# # General CallbackQuery Handlers (fallback)
    application.add_handler(CallbackQueryHandler(gridbutton_handler))
    application.add_handler(CallbackQueryHandler(handle_callback_query))

# General Message Handlers (fallback for text inputs)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_input))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, capture_mint_address))


   # Register the global error handler
    application.add_error_handler(error_handler)

    my_commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("settings", "Open settings to set analysis period"),
        BotCommand("help", "Get help and usage instructions")
    ]
    application.bot.set_my_commands(my_commands)
    
    application.run_polling()
