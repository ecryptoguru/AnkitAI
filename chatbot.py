import os
import sys
import time
import requests

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import CDP Agentkit Langchain Extension.
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.tools import CdpTool
from pydantic import BaseModel, Field
from cdp import Wallet

from twitter_langchain import (TwitterApiWrapper, TwitterToolkit)

MORALIS_API_KEY = os.environ.get("MORALIS_API_KEY")

# Configure a file to persist the agent's CDP MPC Wallet Data.
wallet_data_file = "wallet_data.txt"

# Existing Multi-Token Deployment Prompt
DEPLOY_MULTITOKEN_PROMPT = """
This tool deploys a new multi-token contract with a specified base URI for token metadata.
The base URI should be a template URL containing {id} which will be replaced with the token ID.
For example: 'https://example.com/metadata/{id}.json'
"""

# Token Metadata Prompts
TOKEN_METADATA_PROMPT = """
Fetch metadata for an ERC-20 token using the Moralis API. 
Provides comprehensive information about a specific token, 
including name, symbol, decimals, total supply, and verification status.
"""

TOKEN_DETAILS_PROMPT = """
Fetch comprehensive details about a specific ERC-20 token on the Base blockchain.
Provides in-depth information including price, market cap, security score, 
holders change, volume changes, and price performance.
"""

WALLET_TOKENS_PROMPT = """
Fetch the list of ERC-20 tokens held by the agent's wallet using the Moralis API. 
This action retrieves token balances, contract details, and optional USD price information.
"""

TOKEN_PAIRS_PROMPT = """
Retrieve trading pairs for a specific ERC-20 token on the Base blockchain.
Returns detailed information about token trading pairs, including liquidity, 
price, and exchange details.
"""

TRENDING_TOKENS_PROMPT = """
Discover trending tokens on the Base blockchain with optional 
filtering by security score and market capitalization.
Provides comprehensive information about top-performing tokens using the Moralis API.
"""

WALLET_PNL_PROMPT = """
Calculate and retrieve Profit and Loss (PnL) information for the agent's wallet assets.
Provides detailed insights into token investments, realized profits, and average buy prices.
"""

# Input Models
class DeployMultiTokenInput(BaseModel):
    """Input argument schema for deploy multi-token contract action."""
    base_uri: str = Field(
        ...,
        description="The base URI template for token metadata. Must contain {id} placeholder.",
        example="https://example.com/metadata/{id}.json")

class TokenMetadataInput(BaseModel):
    token_address: str = Field(
        ..., 
        description="Contract address of the ERC-20 token",
        example="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    )

class WalletTokensInput(BaseModel):
    chain: str = Field(
        ...,
        description="Wallet address",
        example="0x0dc74cabcfb00ab5fdeef60088685a71fef97003"
    )

class TokenDetailsInput(BaseModel):

    token_address: str = Field(
        ...,
        description="The contract address of the ERC-20 token to retrieve details for",
        example="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    )

class WalletNftsInput(BaseModel):
    """Input argument schema for get wallet NFTs action."""

    token_address: str = Field(
        ..., 
        description="The wallet address to retrieve NFTs for",
        example="0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    )


class TokenPairsInput(BaseModel):
    token_address: str = Field(
        ..., 
        description="Contract address of the ERC-20 token to find trading pairs",
        example="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    )

class TrendingTokensInput(BaseModel):
    """Input argument schema for fetching trending tokens."""
    security_score: int = Field(
        default=80,
        description="Minimum security score for tokens",
        ge=0,
        le=100
    )
    min_market_cap: int = Field(
        default=100000,
        description="Minimum market cap for tokens",
        ge=0
    )

# Function definitions
def deploy_multi_token(wallet: Wallet, base_uri: str) -> str:
    """Deploy a new multi-token contract with the specified base URI."""
    if "{id}" not in base_uri:
        raise ValueError("base_uri must contain {id} placeholder")

    deployed_contract = wallet.deploy_multi_token(base_uri)
    result = deployed_contract.wait()

    return f"Successfully deployed multi-token contract at address:{result.contract_address}"

def get_token_metadata(token_address: str) -> str:
    """
    Fetch metadata for an ERC-20 token using the Moralis API.
    Automatically determines if the network is mainnet or testnet.

    Args:
        token_address (str): The address of the ERC-20 token

    Returns:
        str: A message with the token metadata or an error message if unsuccessful
    """
    # Read the Moralis API key from the environment
    if not MORALIS_API_KEY:
        return "Error: Moralis API key is missing. Please set the MORALIS_API_KEY environment variable."

    # Determine the network dynamically based on the agent's current network ID
    is_mainnet = Wallet.network_id in ["base", "base-mainnet"]
    chain = "base" if is_mainnet else "base sepolia"

    # API endpoint and headers
    url = "https://deep-index.moralis.io/api/v2.2/erc20/metadata"
    headers = {
        "accept": "application/json",
        "X-API-Key": MORALIS_API_KEY
    }
    params = {
        "chain": chain,
        "addresses[0]": token_address
    }

    # Fetch token metadata
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        metadata = response.json()

        if metadata:
            token_data = metadata[0]
            return (
                f"Token Name: {token_data.get('name')}\n"
                f"Symbol: {token_data.get('symbol')}\n"
                f"Decimals: {token_data.get('decimals')}\n"
                f"Total Supply: {token_data.get('total_supply_formatted')}\n"
                f"Contract Address: {token_data.get('address')}\n"
                f"Verified: {token_data.get('verified_contract')}\n"
                f"Logo URL: {token_data.get('logo')}\n"
            )
        else:
            return "No metadata found for the provided token address."

    except requests.exceptions.RequestException as e:
        return f"Error fetching token metadata: {str(e)}"

def get_wallet_tokens(token_address: str) -> str:
    """
    Fetch the list of ERC-20 tokens held by the agent's wallet using the Moralis API.

    Returns:
        str: A message with the list of tokens and balances or an error message if unsuccessful
    """
    # Get the agent's wallet address
    address_id = token_address

    # Determine the network dynamically based on the agent's current network ID
    is_mainnet = Wallet.network_id in ["base", "base-mainnet"]
    chain = "base" if is_mainnet else "base sepolia"

    # API endpoint and headers
    url = f"https://deep-index.moralis.io/api/v2.2/wallets/{address_id}/tokens"
    headers = {
        "accept": "application/json",
        "X-API-Key": MORALIS_API_KEY
    }
    params = {
        "chain": chain
    }

    # Fetch wallet token balances
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        tokens = response.json().get("result", [])

        # Format the output
        if tokens:
            token_list = "\n".join(
                [
                    f"Token: {token['name']} ({token['symbol']})\n"
                    f"Balance: {token['balance_formatted']} {token['symbol']}\n"
                    f"Contract Address: {token['token_address']}\n"
                    f"Verified: {'Yes' if token['verified_contract'] else 'No'}\n"
                    f"Price (USD): {token['usd_price'] or 'N/A'}\n"
                    for token in tokens
                ]
            )
            return f"Tokens held by {address_id}:\n{token_list}"
        else:
            return f"No tokens found for wallet {address_id}."

    except requests.exceptions.RequestException as e:
        return f"Error fetching wallet tokens: {str(e)}"

def get_token_details(token_address: str) -> str:
    """
    Fetch detailed information about a specific ERC-20 token on the Base blockchain using MOralis API
    Automatically determines if the network is mainnet or testnet.

    Args:
        token_address (str): The address of the ERC-20 token.

    Returns:
        str: Information about the token or an error message if unsuccessful.
    """
    # Determine the network dynamically based on the agent's current network ID
    is_mainnet = Wallet.network_id in ["base", "base-mainnet"]
    chain = "base" if is_mainnet else "base sepolia"

    # API endpoint and headers
    url = "https://deep-index.moralis.io/api/v2.2/discovery/token"
    headers = {
        "accept": "application/json",
        "X-API-Key": MORALIS_API_KEY
    }
    params = {
        "chain": chain,
        "token_address": token_address
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        token_data = response.json()

        # Format the output
        token_info = (
            f"Token Name: {token_data.get('token_name')}\n"
            f"Symbol: {token_data.get('token_symbol')}\n"
            f"Price (USD): {token_data.get('price_usd')}\n"
            f"Market Cap: {token_data.get('market_cap')}\n"
            f"Security Score: {token_data.get('security_score')}\n"
            f"Token Age (days): {token_data.get('token_age_in_days')}\n"
            f"On-Chain Strength Index: {token_data.get('on_chain_strength_index')}\n"
            f"1-Day Holders Change: {token_data['holders_change'].get('1d')}\n"
            f"1-Day Volume Change (USD): {token_data['volume_change_usd'].get('1d')}\n"
            f"1-Month Price Change (%): {token_data['price_percent_change_usd'].get('1M')}\n"
            f"Logo: {token_data.get('token_logo')}\n"
        )
        return token_info

    except requests.exceptions.RequestException as e:
        return f"Error fetching token details: {str(e)}"

def get_wallet_nfts(token_address: str) -> str:
    """
    Fetch the raw response of NFTs held by the agent's wallet on the Base blockchain.
    Automatically determines if the network is mainnet or testnet.

    Returns:
        str: Raw JSON response of NFTs or an error message if unsuccessful.
    """
    # Get the agent's wallet address
    wallet_address = token_address

    # Determine the network dynamically based on the agent's current network ID
    is_mainnet = Wallet.network_id in ["base", "base-mainnet"]
    chain = "base" if is_mainnet else "base sepolia"

    # API endpoint and headers
    url = f"https://deep-index.moralis.io/api/v2.2/{wallet_address}/nft"
    headers = {
        "accept": "application/json",
        "X-API-Key": MORALIS_API_KEY
    }
    params = {
        "chain": chain,
        "format": "decimal",
        "media_items": "false"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.text  # Return the raw JSON response as text

    except requests.exceptions.RequestException as e:
        return f"Error fetching wallet NFTs: {str(e)}"

                
def get_token_pairs(token_address: str) -> str:
            """
            Fetch trading pairs for a specific ERC-20 token on the Base blockchain.
            Automatically determines if the network is mainnet or testnet.

            Args:
                token_address (str): The address of the ERC-20 token.

            Returns:
                str: Information about trading pairs or an error message if unsuccessful.
            """
            # Determine the network dynamically based on the agent's current network ID
            is_mainnet = Wallet.network_id in ["base", "base-mainnet"]
            chain = "base" if is_mainnet else "base sepolia"

            # API endpoint and headers
            url = f"https://deep-index.moralis.io/api/v2.2/erc20/{token_address}/pairs"
            headers = {
                "accept": "application/json",
                "X-API-Key": MORALIS_API_KEY
            }
            params = {
                "chain": chain
            }

            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                pairs = response.json().get("pairs", [])

                # Format the output
                if pairs:
                    pairs_info = "\n".join(
                        [
                            f"Pair: {pair['pair_label']}\n"
                            f"Price (USD): {pair['usd_price']}\n"
                            f"24hr Price Change (%): {pair['usd_price_24hr_percent_change']}\n"
                            f"Liquidity (USD): {pair['liquidity_usd']}\n"
                            f"Exchange Address: {pair['exchange_address']}\n"
                            f"Base Token: {pair['pair'][0]['token_name']} ({pair['pair'][0]['token_symbol']})\n"
                            f"Quote Token: {pair['pair'][1]['token_name']} ({pair['pair'][1]['token_symbol']})\n"
                            for pair in pairs
                        ]
                    )
                    return f"Trading pairs for token {token_address}:\n{pairs_info}"
                else:
                    return f"No trading pairs found for token {token_address}."

            except requests.exceptions.RequestException as e:
                return f"Error fetching token pairs: {str(e)}"


def get_trending_tokens(security_score=80, min_market_cap=100000) -> str:
        """
        Fetch trending tokens with a minimum security score and market cap.

        Args:
            security_score (int): Minimum security score for tokens
            min_market_cap (int): Minimum market cap for tokens

        Returns:
            str: Trending token information or an error message
        """
        url = "https://deep-index.moralis.io/api/v2.2/discovery/tokens/trending"
        headers = {
            "accept": "application/json",
            "X-API-Key": MORALIS_API_KEY
        }
        params = {
            "chain": "base",
            "security_score": security_score,
            "min_market_cap": min_market_cap
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            tokens = response.json()

            # Format the output
            token_info = "\n".join(
                [
                    f"Token Name: {token['token_name']} ({token['token_symbol']})\n"
                    f"Price (USD): {token['price_usd']}\n"
                    f"Market Cap: {token['market_cap']}\n"
                    f"Security Score: {token['security_score']}\n"
                    f"Logo: {token['token_logo']}\n"
                    for token in tokens
                ]
            )
            return f"Trending Tokens:\n{token_info}"

        except requests.exceptions.RequestException as e:
            return f"Error fetching trending tokens: {str(e)}"
        

def initialize_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatOpenAI(model="gpt-4o-mini")

    wallet_data = None

    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit     Wrapper.
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # persist the agent's CDP MPC Wallet Data.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()
    twitter_api_wrapper = TwitterApiWrapper()
    twitter_toolkit = TwitterToolkit.from_twitter_api_wrapper(
        twitter_api_wrapper)
    tools.extend(twitter_toolkit.get_tools())

    # Existing Multi-Token Deployment Tool
    deployMultiTokenTool = CdpTool(
        name="deploy_multi_token",
        description=DEPLOY_MULTITOKEN_PROMPT,
        cdp_agentkit_wrapper=agentkit,
        args_schema=DeployMultiTokenInput,
        func=deploy_multi_token,
    )

    # Token Metadata Tool
    tokenMetadataTool = CdpTool(
        name="get_token_metadata",
        description=TOKEN_METADATA_PROMPT,
        cdp_agentkit_wrapper=agentkit,
        args_schema=TokenMetadataInput,
        func=get_token_metadata,
    )

    # Get Wallet Tokens Tool
    walletTokensTool = CdpTool(
        name="get_wallet_tokens",
        description=WALLET_TOKENS_PROMPT,
        cdp_agentkit_wrapper=agentkit,
        args_schema=WalletTokensInput,
        func=get_wallet_tokens,
    )

    # Get Token Details Tool
    tokenDetailsTool = CdpTool(
        name="get_token_details",
        description="""
        This tool fetches detailed information about an ERC-20 token on the Base blockchain, 
        including key metrics like token name, symbol, price, market cap, security score, 
        and historical performance indicators using the Moralis API.
        """,
        cdp_agentkit_wrapper=agentkit,
        args_schema=TokenDetailsInput,
        func=get_token_details,
    )

    # Get Wallet NFTs Tool
    walletNftsTool = CdpTool(
        name="get_wallet_nfts",
        description="""Fetch the raw response of NFTs held by a wallet on the Base blockchain. 
        This action retrieves NFT information using the Moralis API, automatically 
        determining the correct network (mainnet or testnet) based on the wallet's network ID.
        """,
        cdp_agentkit_wrapper=agentkit,
        args_schema=WalletNftsInput,
        func=get_wallet_nfts,
    )
    
    # Token Pairs Tool
    tokenPairsTool = CdpTool(
        name="get_token_pairs",
        description=TOKEN_PAIRS_PROMPT,
        cdp_agentkit_wrapper=agentkit,
        args_schema=TokenPairsInput,
        func=get_token_pairs,
    )

    # Trending Tokens Tool
    trendingTokensTool = CdpTool(
        name="get_trending_tokens",
        description=TRENDING_TOKENS_PROMPT,
        cdp_agentkit_wrapper=agentkit,
        args_schema=TrendingTokensInput,
        func=get_trending_tokens,
    )

    


    # Add all tools to the tools list
    tools.extend([deployMultiTokenTool, tokenMetadataTool, tokenPairsTool, walletTokensTool, tokenDetailsTool, walletNftsTool, trendingTokensTool])

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=
        "You are a helpful agent that can interact onchain using the Coinbase                  Developer Platform Agentkit. You are empowered to interact onchain using your          tools. If you ever need funds, you can request them from the faucet if you are         on network ID `base-sepolia`. If not, you can provide your wallet details and          request funds from the user. If someone asks you to do something you can't do          with your currently available tools, you must say so, and encourage them to            implement it themselves using the CDP SDK + Agentkit, recommend they go to             docs.cdp.coinbase.com for more informaton. Be concise and helpful with your            responses. Refrain from restating your tools' descriptions unless it is                explicitly requested. You are a specialized investment agent on the Base Layer 2 blockchain, designed to optimize an existing portfolio by analyzing and trading trending tokens.Your primary goal is to identify profitable tokens in the market, review wallet balances, and make calculated swap decisions to enhance the portfolio value.Follow these steps when making investment decisions: 1. Use trending data to identify promising tokens with potential profit.2. For each trending token, retrieve detailed information to evaluate its market cap, liquidity, and security. 3. Check the wallet balance to understand the available assets and decide on a safe percentage to invest. 4. Execute swaps to acquire trending tokens, ensuring the chosen amount aligns with profitability goals and balance management. Make data-driven decisions based on token performance, wallet balance, and profitability, while maximizing portfolio value with each trade. Use all available functions to analyze market trends, asset details, and wallet metrics to act with precision and efficiency"
    ), config

# Autonomous Mode
def run_autonomous_mode(agent_executor, config, interval=10):
    """Run the agent autonomously with specified intervals."""
    print("Starting autonomous mode...")
    while True:
        try:
            # Provide instructions autonomously
            thought = (
                "Be creative and do something interesting on the blockchain. "
                "Choose an action or set of actions and execute it that highlights                      your abilities."
            )

            # Run agent in autonomous mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=thought)]}, config):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

            # Wait before the next action
            time.sleep(interval)

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


# Chat Mode
def run_chat_mode(agent_executor, config):
    """Run the agent interactively based on user input."""
    print("Starting chat mode... Type 'exit' to end.")
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() == "exit":
                break

            # Run agent with the user's input in chat mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}, config):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


# Mode Selection
def choose_mode():
    """Choose whether to run in autonomous or chat mode based on user input."""
    while True:
        print("\nAvailable modes:")
        print("1. chat    - Interactive chat mode")
        print("2. auto    - Autonomous action mode")

        choice = input(
            "\nChoose a mode (enter number or name): ").lower().strip()
        if choice in ["1", "chat"]:
            return "chat"
        elif choice in ["2", "auto"]:
            return "auto"
        print("Invalid choice. Please try again.")


def main():
    """Start the chatbot agent."""
    agent_executor, config = initialize_agent()

    mode = choose_mode()
    if mode == "chat":
        run_chat_mode(agent_executor=agent_executor, config=config)
    elif mode == "auto":
        run_autonomous_mode(agent_executor=agent_executor, config=config)


if __name__ == "__main__":
    print("Starting Agent...")
    main()

