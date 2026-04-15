This project was developed in collaboration with https://github.com/tanh2-cz/ai_debate.

# Multi-Role AI Debate Platform

Designed for debate simulation and training.
A multi-role AI debate system based on Streamlit, supporting 3-6 different professional roles to conduct intelligent debates on specific topics.
It integrates the DeepSeek large language model and Kimi's web search functionality to provide real-time academic resource support for each role.

**Update:** This version supports a full four-stage debate process, including: opening statements, Q&A session, free debate, and closing statements. During the Q&A session, each agent has the opportunity to ask questions to any other agent. The free debate session supports customizing the number of speaking rounds.

## Main Features

- **Multi-Role Debate**: Supports 6 professional roles: Environmentalist, Economist, Policymaker, Technology Expert, Sociologist, and Ethicist.
- **Intelligent Web Search**: Integrates the Kimi API to provide the latest academic materials and research reports for each role.
- **Real-time Debate Display**: Streams the debate process, supporting multi-round in-depth discussions.
- **Caching Mechanism**: Smartly caches search results to improve response speed.
- **Flexible Configuration**: Users can customize participating roles, debate rounds, the number of references, and other parameters.

## Requirements

- DeepSeek API Key
- Kimi API Key

## Installation Steps

Create a `.env` file based on `.env.example` and add the following content:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key
KIMI_API_KEY=your_kimi_api_key
   ```

## Usage

1. **Launch the app**
   ```bash
   conda create -n debate python=3.10
   conda activate debate
   pip install -r requirements.txt
   streamlit run debates.py
   ```

2. **Configure the debate**
   - Select 3–6 participant roles in the sidebar
   - Set whether to enable Kimi web search
   - Configure the number of references for each role (1–5)
   - Set the number of rounds for the free debate phase (2–8 rounds)

3. **Choose a topic**
   - Select from preset topics, or define a custom debate topic
   - The topic can be any controversial real-world issue

4. **Start the debate**
   - Click the **"Start Debate"** button
   - The system will automatically search for relevant sources for each role (if enabled)
   - Watch the real-time debate process among AI experts

## Configuration Guide

### How to Get API Keys

- **DeepSeek API**: Visit [DeepSeek Platform](https://platform.deepseek.com/) to register and obtain your key
- **Kimi API**: Visit [Moonshot AI](https://www.moonshot.cn/) to register and obtain your key

### Role Descriptions

| Role | Domain Expertise | Focus Area |
|------|------------------|------------|
| Environmentalist | Environmental Science | Ecological balance and sustainable development |
| Economist | Market Economics | Cost-effectiveness and market mechanisms |
| Policymaker | Public Administration | Policy feasibility and social governance |
| Technology Expert | R&D / Technology | Technological innovation and implementation pathways |
| Sociologist | Social Research | Social impact and human-centered concerns |
| Ethicist | Moral Philosophy | Ethics, morality, and value judgments |

## Project Structure

```text
├── debates.py          # Main application file (Streamlit interface)
├── graph.py            # Multi-agent debate logic
├── rag_module.py       # Kimi web search module
├── requirements.txt    # Required package list
├── .env                # Environment variable configuration
└── README.md           # Project documentation
```
