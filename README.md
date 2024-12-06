# CDP Agentkit Chatbot - Replit Quickstart Template

A Replit template for running an AI agent with onchain capabilities using the [Coinbase Developer Platform (CDP) Agentkit](https://github.com/coinbase/cdp-agentkit/).

This template demonstrates a terminal-based chatbot with full access to CDP Agentkit actions, allowing it to perform blockchain operations like:
- Deploying tokens (ERC-20 & NFTs)
- Managing wallets
- Executing transactions
- Interacting with smart contracts

## Prerequisites

1. **Replit Core Subscription**
   - Reccomended for deploying the bot.
   - For Coinbase Community sponsorship of Replit Core, send an email to: kevin.leffew@coinbase.com

2. **API Keys**
   - OpenAI API key from the [OpenAI Portal](https://platform.openai.com/api-keys)
   - CDP API credentials from [CDP Portal](https://portal.cdp.coinbase.com/access/api)

## Quick Start

1. **Configure Secrets and CDP API Keys**
   Navigate to Tools > Secrets and add the secrets above

2. **Run the Bot**
- Click the Run button
- Choose between chat mode or autonomous mode
- Start interacting onchain!

## Securing your Wallets

Every agent comes with an associated wallet. Wallet data is read from wallet_data.txt, and if that file does not exist, this repl will create a new wallet and persist it in a new file. Please note that this contains sensitive data and should not be used in production environments. Refer to the [CDP docs](https://docs.cdp.coinbase.com/mpc-wallet/docs/wallets#securing-a-wallet) for information on how to secure your wallets.

## Features
- Interactive chat mode for guided interactions
- Autonomous mode for self-directed blockchain operations
- Full CDP Agentkit integration
- Persistent wallet management

## Source
This template is based on the CDP Agentkit examples. For more information, visit:
https://github.com/coinbase/cdp-agentkit