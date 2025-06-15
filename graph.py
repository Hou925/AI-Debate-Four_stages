"""
多角色AI辩论系统核心逻辑 - Kimi联网搜索集成版本（四阶段辩论版）
支持3-6个不同角色的智能辩论，基于Kimi API的联网搜索功能
四阶段辩论：开辩综述 -> 提问回答 -> 自由辩论 -> 结辩综述
"""

from typing import TypedDict, Literal, List, Dict, Any, Optional
import os
from dotenv import find_dotenv, load_dotenv
import random

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

# 导入基于Kimi联网搜索的RAG模块
from rag_module import initialize_rag_module, get_rag_module, DynamicRAGModule

# 加载环境变量
load_dotenv(find_dotenv())

# 全局变量
deepseek = None
rag_module = None

# 初始化DeepSeek模型和基于Kimi联网搜索的RAG模块
try:
    deepseek = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.8,        # 稍微提高温度增加观点多样性
        max_tokens=2000,        # 增加token限制以容纳联网搜索内容
        timeout=60,
        max_retries=3,
    )
    print("✅ DeepSeek模型初始化成功")
    
    # 初始化基于Kimi联网搜索的RAG模块
    rag_module = initialize_rag_module(deepseek)
    if rag_module:
        print("✅ Kimi联网搜索模块初始化成功")
    else:
        print("⚠️ Kimi联网搜索模块初始化失败，将使用传统模式")
    
except Exception as e:
    print(f"❌ 模型初始化失败: {e}")
    deepseek = None
    rag_module = None


class MultiAgentDebateState(MessagesState):
    """多角色辩论状态管理（四阶段版）"""
    main_topic: str = "人工智能的发展前景"
    
    # 辩论阶段相关
    current_stage: str = "opening"          # opening, questioning, free_debate, closing
    stage_progress: int = 0                 # 当前阶段进度
    max_rounds: int = 3                     # 自由辩论阶段的最大轮次
    
    # Agent相关
    active_agents: List[str] = []           # 活跃的Agent列表
    current_agent_index: int = 0            # 当前发言Agent索引
    total_messages: int = 0                 # 总消息数
    
    # RAG配置
    rag_enabled: bool = True                # RAG功能开关
    rag_sources: List[str] = ["web_search"] # RAG数据源（联网搜索）
    collected_references: List[Dict] = []   # 收集的参考文献
    max_refs_per_agent: int = 3             # 每个专家的最大参考文献数
    max_results_per_source: int = 2         # 每个数据源的最大检索数
    
    # 专家缓存相关
    agent_paper_cache: Dict[str, str] = {}  # 格式: {agent_key: rag_context}
    first_round_rag_completed: List[str] = []  # 已完成第一轮RAG检索的专家列表
    
    # 提问阶段相关
    questions_asked: List[Dict] = []        # 记录已提出的问题 [{questioner, target, question, answer}]
    current_questioner: str = ""           # 当前提问者
    current_target: str = ""               # 当前被提问者
    waiting_for_answer: bool = False       # 是否等待回答
    
    # 阶段记录
    opening_statements: Dict[str, str] = {}  # 开辩陈述
    closing_statements: Dict[str, str] = {}  # 结辩陈述
    
    # 简化的状态字段
    agent_positions: Dict[str, List[str]] = {}  # 基本的专家立场记录
    key_points_raised: List[str] = []       # 基本的关键论点
    controversial_points: List[str] = []    # 基本的争议观点


# 定义所有可用的角色（保持不变）
AVAILABLE_ROLES = {
    "environmentalist": {
        "name": "环保主义者",
        "role": "环境保护倡导者",
        "icon": "🌱",
        "color": "#4CAF50",
        "focus": "生态平衡与可持续发展",
        "perspective": "任何决策都应考虑对环境的长远影响",
        "bio": "专业的环境保护主义者，拥有环境科学博士学位。长期关注气候变化、生物多样性保护和可持续发展。坚信经济发展必须与环境保护相协调，主张采用清洁技术和循环经济模式。",
        "speaking_style": "理性分析环境数据，引用科学研究，强调长期后果",
        "search_keywords": "环境保护 气候变化 可持续发展 生态影响 环境科学"
    },
    
    "economist": {
        "name": "经济学家", 
        "role": "市场经济分析专家",
        "icon": "📊",
        "color": "#FF9800",
        "focus": "成本效益与市场机制",
        "perspective": "追求经济效率和市场最优解决方案",
        "bio": "资深经济学教授，专攻宏观经济学和政策分析。擅长成本效益分析、市场失灵研究和经济政策评估。相信市场机制的力量，但也认识到政府干预的必要性。",
        "speaking_style": "用数据说话，分析成本收益，关注市场效率和经济可行性",
        "search_keywords": "经济影响 成本效益 市场分析 经济政策 宏观经济"
    },
    
    "policy_maker": {
        "name": "政策制定者",
        "role": "公共政策专家", 
        "icon": "🏛️",
        "color": "#3F51B5",
        "focus": "政策可行性与社会治理",
        "perspective": "平衡各方利益，制定可执行的政策",
        "bio": "资深公务员和政策分析师，拥有公共管理硕士学位。在政府部门工作多年，熟悉政策制定流程、法律法规和实施挑战。善于协调各方利益，寻求平衡解决方案。",
        "speaking_style": "考虑实施难度，关注法律框架，寻求各方共识",
        "search_keywords": "政策制定 监管措施 治理框架 实施策略 公共政策"
    },
    
    "tech_expert": {
        "name": "技术专家",
        "role": "前沿科技研究者",
        "icon": "💻", 
        "color": "#9C27B0",
        "focus": "技术创新与实现路径",
        "perspective": "技术进步是解决问题的关键驱动力",
        "bio": "计算机科学博士，在科技公司担任首席技术官。专注于人工智能、机器学习和新兴技术研发。相信技术创新能够解决人类面临的重大挑战，但也关注技术伦理问题。",
        "speaking_style": "分析技术可行性，讨论创新解决方案，关注实现路径",
        "search_keywords": "技术创新 技术可行性 技术发展 技术影响 前沿科技"
    },
    
    "sociologist": {
        "name": "社会学家",
        "role": "社会影响研究专家", 
        "icon": "👥",
        "color": "#E91E63",
        "focus": "社会影响与人文关怀",
        "perspective": "关注对不同社会群体的影响和社会公平",
        "bio": "社会学教授，专注于社会变迁、不平等研究和社会政策分析。长期关注技术变革对社会结构的影响，特别是对弱势群体的影响。主张包容性发展和社会公正。",
        "speaking_style": "关注社会公平，分析对不同群体的影响，强调人文关怀",
        "search_keywords": "社会影响 社会变化 社群效应 社会公平 社会学研究"
    },
    
    "ethicist": {
        "name": "伦理学家",
        "role": "道德哲学研究者",
        "icon": "⚖️", 
        "color": "#607D8B",
        "focus": "伦理道德与价值判断",
        "perspective": "坚持道德原则和伦理标准",
        "bio": "哲学博士，专攻应用伦理学和技术伦理。在大学教授道德哲学，并为政府和企业提供伦理咨询。关注新技术带来的伦理挑战，主张在发展中坚持道德底线。",
        "speaking_style": "引用伦理原则，分析道德后果，坚持价值标准",
        "search_keywords": "伦理道德 道德责任 价值观念 伦理框架 道德哲学"
    }
}


# 四阶段辩论提示词模板

# 开辩阶段模板
OPENING_STATEMENT_TEMPLATE = """
你是一位{role} - {name}。

【角色背景】
{bio}

【你的专业视角】
- 关注重点：{focus}
- 核心观点：{perspective}
- 表达风格：{speaking_style}

【辩论信息】
辩论主题：{main_topic}
当前阶段：开辩综述阶段
你的发言顺序：第 {agent_position} 位发言
参与者：{other_participants}

【基于联网搜索的最新资料】
{rag_context}

【任务要求】
这是开辩综述阶段，请作为{name}，针对辩论主题"{main_topic}"发表你的开场陈述：

1. **明确立场**：清晰表达你对这个议题的基本观点和立场
2. **核心论点**：提出2-3个支撑你观点的主要论据
3. **专业视角**：充分体现你作为{role}的专业特色
4. **引用资料**：适当引用联网搜索获得的最新信息和数据
5. **逻辑清晰**：确保论述条理分明，逻辑严密

【发言要求】
- 控制在4-5句话内，确保内容充实而简洁
- 语气要体现专业性和权威性
- 为后续的提问和辩论环节铺垫

现在请发表你的开辩陈述：
"""

# 提问阶段模板
QUESTIONING_TEMPLATE = """
你是一位{role} - {name}。

【角色背景】
{bio}

【当前辩论情况】
辩论主题：{main_topic}
当前阶段：提问回答阶段
参与者：{other_participants}

【已完成的开辩陈述】
{opening_statements}

【基于联网搜索的最新资料】
{rag_context}

【任务要求】
{task_description}

【发言要求】
- 控制在3-4句话内
- 保持专业性和针对性
- {specific_instructions}

现在请{action_type}：
"""

# 自由辩论阶段模板
FREE_DEBATE_TEMPLATE = """
你是一位{role} - {name}。

【角色背景】
{bio}

【当前辩论情况】
辩论主题：{main_topic}
当前阶段：自由辩论阶段
当前轮次：第 {current_round} 轮（共 {max_rounds} 轮）
你的发言顺序：第 {agent_position} 位
参与者：{other_participants}

【前期重要内容回顾】
开辩陈述要点：
{opening_summary}

关键提问与回答：
{qa_summary}

【基于联网搜索的最新资料】
{rag_context}

【最近对话历史】
{history}

【任务要求】
在自由辩论阶段，请针对其他专家的观点进行回应和辩论：

1. **回应观点**：对其他专家刚才的发言进行回应
2. **深化论证**：进一步阐述和强化你的观点
3. **反驳质疑**：对你认为有问题的观点进行理性反驳
4. **寻求共识**：在分歧中寻找可能的共同点
5. **专业特色**：始终保持你的专业角色特色

【发言要求】
- 控制在3-4句话内
- 保持理性和专业
- 既要坚持立场又要开放对话

现在请在第{current_round}轮自由辩论中发言：
"""

# 结辩阶段模板
CLOSING_STATEMENT_TEMPLATE = """
你是一位{role} - {name}。

【角色背景】
{bio}

【辩论主题】
{main_topic}

【你在整场辩论中的核心观点】
{your_position_summary}

【其他专家的主要观点】
{others_positions_summary}

【整场辩论的关键争议点】
{key_controversies}

【基于联网搜索的最新资料】
{rag_context}

【任务要求】
这是结辩综述阶段，请作为{name}发表你的结束陈词：

1. **重申立场**：简明扼要地重申你的核心观点
2. **总结论据**：总结你在整场辩论中提出的最有力论据
3. **回应挑战**：简要回应其他专家对你观点的主要挑战
4. **呼吁行动**：基于你的专业角色，提出具体的建议或呼吁
5. **展望未来**：对这个议题的未来发展提出你的专业看法

【发言要求】
- 控制在4-5句话内
- 语气要有总结性和前瞻性
- 体现你作为{role}的专业权威性
- 给整场辩论一个有力的收尾

现在请发表你的结辩陈词：
"""


def create_opening_chat_template():
    """创建开辩阶段聊天模板"""
    return ChatPromptTemplate.from_messages([
        ("system", OPENING_STATEMENT_TEMPLATE),
        ("user", "请发表你的开辩综述"),
    ])


def create_questioning_chat_template():
    """创建提问阶段聊天模板"""
    return ChatPromptTemplate.from_messages([
        ("system", QUESTIONING_TEMPLATE),
        ("user", "请按要求执行"),
    ])


def create_free_debate_chat_template():
    """创建自由辩论阶段聊天模板"""
    return ChatPromptTemplate.from_messages([
        ("system", FREE_DEBATE_TEMPLATE),
        ("user", "请参与自由辩论"),
    ])


def create_closing_chat_template():
    """创建结辩阶段聊天模板"""
    return ChatPromptTemplate.from_messages([
        ("system", CLOSING_STATEMENT_TEMPLATE),
        ("user", "请发表你的结辩陈词"),
    ])


def format_opening_statements(opening_statements: Dict[str, str], active_agents: List[str]) -> str:
    """格式化开辩陈述"""
    if not opening_statements:
        return "暂无开辩陈述。"
    
    formatted = []
    for agent_key in active_agents:
        if agent_key in opening_statements:
            agent_name = AVAILABLE_ROLES[agent_key]["name"]
            statement = opening_statements[agent_key]
            # 清理陈述内容
            clean_statement = statement.replace(f"{agent_name}:", "").strip()
            formatted.append(f"{agent_name}: {clean_statement}")
    
    return "\n\n".join(formatted)


def format_qa_summary(questions_asked: List[Dict]) -> str:
    """格式化提问回答摘要"""
    if not questions_asked:
        return "暂无提问回答记录。"
    
    formatted = []
    for i, qa in enumerate(questions_asked, 1):
        questioner_name = AVAILABLE_ROLES[qa["questioner"]]["name"]
        target_name = AVAILABLE_ROLES[qa["target"]]["name"]
        formatted.append(f"Q{i}: {questioner_name} → {target_name}")
        formatted.append(f"问题: {qa['question']}")
        formatted.append(f"回答: {qa['answer']}")
        formatted.append("")
    
    return "\n".join(formatted)


def format_agent_history(messages: List, active_agents: List[str], current_agent: str, current_round: int, max_show: int = 6) -> str:
    """格式化对话历史（自由辩论阶段）"""
    if not messages:
        return "这是自由辩论的开始。"
    
    formatted_history = []
    
    # 显示最近的消息
    recent_messages = messages[-max_show:] if len(messages) > max_show else messages
    
    for i, message in enumerate(recent_messages):
        # 估算发言者（这是一个简化的估算）
        global_msg_idx = len(messages) - len(recent_messages) + i
        agent_index = global_msg_idx % len(active_agents)
        agent_key = active_agents[agent_index]
        agent_name = AVAILABLE_ROLES[agent_key]["name"]
        
        # 获取消息内容
        if hasattr(message, 'content'):
            message_content = message.content
        elif isinstance(message, str):
            message_content = message
        else:
            message_content = str(message)
        
        # 清理消息内容
        clean_message = message_content.replace(f"{agent_name}:", "").strip()
        formatted_history.append(f"{agent_name}: {clean_message}")
    
    return "\n\n".join(formatted_history)


def get_other_participants(active_agents: List[str], current_agent: str) -> str:
    """获取其他参与者信息"""
    others = []
    for agent_key in active_agents:
        if agent_key != current_agent:
            agent_info = AVAILABLE_ROLES[agent_key]
            others.append(f"- {agent_info['name']}({agent_info['role']})")
    return "\n".join(others)


def get_rag_context_for_agent(agent_key: str, debate_topic: str, state: MultiAgentDebateState) -> str:
    """为Agent获取RAG上下文（四阶段版）"""
    # 检查RAG是否启用
    if not state.get("rag_enabled", True) or not rag_module:
        return "当前未启用联网搜索功能。"
    
    # 从状态读取用户设置的参考文献数量
    max_refs_per_agent = state.get("max_refs_per_agent", 3)
    max_results_per_source = state.get("max_results_per_source", 2)
    
    print(f"🔍 为{AVAILABLE_ROLES[agent_key]['name']}进行联网搜索，设置最大文献数为 {max_refs_per_agent} 篇")
    
    # 检查当前阶段
    current_stage = state.get("current_stage", "opening")
    agent_paper_cache = state.get("agent_paper_cache", {})
    first_round_rag_completed = state.get("first_round_rag_completed", [])
    
    try:
        # 如果是开辩阶段且该专家还未搜索过，进行联网搜索并缓存
        if current_stage == "opening" and agent_key not in first_round_rag_completed:
            print(f"🔍 开辩阶段：为{AVAILABLE_ROLES[agent_key]['name']}使用联网搜索...")
            
            context = rag_module.get_rag_context_for_agent(
                agent_role=agent_key,
                debate_topic=debate_topic,
                max_sources=max_refs_per_agent,
                max_results_per_source=max_results_per_source,
                force_refresh=True
            )
            
            # 将结果缓存到状态中
            if context and context.strip() != "暂无相关学术资料。":
                agent_paper_cache[agent_key] = context
                first_round_rag_completed.append(agent_key)
                
                actual_ref_count = context.count('参考资料')
                print(f"✅ 联网搜索成功：{AVAILABLE_ROLES[agent_key]['name']}获得{actual_ref_count}篇资料")
                
                return context
            else:
                print(f"⚠️ {AVAILABLE_ROLES[agent_key]['name']}未找到相关资料")
                return "暂未找到直接相关的最新信息，请基于你的专业知识发表观点。"
        
        # 如果不是开辩阶段或该专家已搜索过，使用缓存
        elif agent_key in agent_paper_cache:
            cached_context = agent_paper_cache[agent_key]
            actual_ref_count = cached_context.count('参考资料')
            print(f"📚 使用缓存：{AVAILABLE_ROLES[agent_key]['name']}获得{actual_ref_count}篇缓存资料")
            return cached_context
        
        # 兜底情况
        else:
            return "暂未找到直接相关的最新信息，请基于你的专业知识发表观点。"
        
    except Exception as e:
        print(f"❌ 获取{agent_key}的联网搜索上下文失败: {e}")
        return "联网搜索遇到技术问题，请基于你的专业知识发表观点。"


def select_next_questioner_and_target(active_agents: List[str], questions_asked: List[Dict]) -> tuple:
    """选择下一个提问者和被提问者"""
    # 统计每个人提问和被提问的次数
    question_count = {agent: 0 for agent in active_agents}
    target_count = {agent: 0 for agent in active_agents}
    
    for qa in questions_asked:
        question_count[qa["questioner"]] += 1
        target_count[qa["target"]] += 1
    
    # 找出提问次数最少的人作为提问者
    min_questions = min(question_count.values())
    candidates_questioner = [agent for agent, count in question_count.items() if count == min_questions]
    questioner = random.choice(candidates_questioner)
    
    # 找出被提问次数最少且不是提问者的人作为被提问者
    available_targets = [agent for agent in active_agents if agent != questioner]
    min_targets = min(target_count[agent] for agent in available_targets)
    candidates_target = [agent for agent in available_targets if target_count[agent] == min_targets]
    target = random.choice(candidates_target)
    
    return questioner, target


def determine_next_node(state: MultiAgentDebateState) -> str:
    """确定下一个节点"""
    current_stage = state.get("current_stage", "opening")
    stage_progress = state.get("stage_progress", 0)
    active_agents = state.get("active_agents", [])
    max_rounds = state.get("max_rounds", 3)
    
    if current_stage == "opening":
        # 开辩阶段：每个人发言一次
        if stage_progress < len(active_agents):
            return active_agents[stage_progress]
        else:
            return "questioning"
    
    elif current_stage == "questioning":
        # 提问阶段：修改这里的逻辑
        questions_asked = state.get("questions_asked", [])
        waiting_for_answer = state.get("waiting_for_answer", False)
        
        if waiting_for_answer:
            # 如果正在等待回答，必须返回被提问者
            current_target = state.get("current_target", "")
            if current_target and current_target in active_agents:
                return current_target
            else:
                # 如果目标无效，重置状态
                return "free_debate"
        else:
            # 检查是否还有人需要提问
            if len(questions_asked) < len(active_agents):
                # 还有人需要提问，选择下一个提问者
                questioner, target = select_next_questioner_and_target(active_agents, questions_asked)
                return questioner
            else:
                # 所有人都提问完了，转到自由辩论
                return "free_debate"
    
    elif current_stage == "free_debate":
        # 自由辩论阶段：轮流发言
        current_round = (stage_progress // len(active_agents)) + 1
        if current_round <= max_rounds:
            agent_index = stage_progress % len(active_agents)
            return active_agents[agent_index]
        else:
            return "closing"
    
    elif current_stage == "closing":
        # 结辩阶段：每个人发言一次
        if stage_progress < len(active_agents):
            return active_agents[stage_progress]
        else:
            return END
    
    return END


def _generate_agent_response(state: MultiAgentDebateState, agent_key: str) -> Dict[str, Any]:
    """生成指定Agent的回复（四阶段版）"""
    if deepseek is None:
        error_msg = f"{AVAILABLE_ROLES[agent_key]['name']}: 抱歉，AI模型未正确初始化。"
        return {
            "messages": [AIMessage(content=error_msg)],
            "total_messages": state.get("total_messages", 0) + 1,
            "stage_progress": state.get("stage_progress", 0) + 1,
        }
    
    try:
        agent_info = AVAILABLE_ROLES[agent_key]
        current_stage = state.get("current_stage", "opening")
        
        # 根据阶段选择不同的模板和处理逻辑
        if current_stage == "opening":
            return _generate_opening_statement(state, agent_key)
        elif current_stage == "questioning":
            if state.get("waiting_for_answer", False) and state.get("current_target", "") == agent_key:
                return _generate_answer(state, agent_key)
            else:
                return _generate_question(state, agent_key)
        elif current_stage == "free_debate":
            return _generate_free_debate_response(state, agent_key)
        elif current_stage == "closing":
            return _generate_closing_statement(state, agent_key)
        else:
            error_msg = f"{agent_info['name']}: 未知的辩论阶段。"
            return {
                "messages": [AIMessage(content=error_msg)],
                "total_messages": state.get("total_messages", 0) + 1,
                "stage_progress": state.get("stage_progress", 0) + 1,
            }
            
    except Exception as e:
        error_msg = f"{AVAILABLE_ROLES[agent_key]['name']}: 抱歉，我现在无法发言。技术问题：{str(e)}"
        print(f"❌ {agent_key} 生成回复时出错: {e}")
        return {
            "messages": [AIMessage(content=error_msg)],
            "total_messages": state.get("total_messages", 0) + 1,
            "stage_progress": state.get("stage_progress", 0) + 1,
        }


def _generate_opening_statement(state: MultiAgentDebateState, agent_key: str) -> Dict[str, Any]:
    """生成开辩陈述"""
    agent_info = AVAILABLE_ROLES[agent_key]
    chat_template = create_opening_chat_template()
    pipe = chat_template | deepseek | StrOutputParser()
    
    # 计算位置信息
    stage_progress = state.get("stage_progress", 0)
    agent_position = stage_progress + 1
    
    # 获取其他参与者信息
    other_participants = get_other_participants(state["active_agents"], agent_key)
    
    # 获取联网搜索上下文
    rag_context = get_rag_context_for_agent(agent_key, state["main_topic"], state)
    
    # 调用模型生成开辩陈述
    response = pipe.invoke({
        "role": agent_info["role"],
        "name": agent_info["name"],
        "bio": agent_info["bio"],
        "focus": agent_info["focus"],
        "perspective": agent_info["perspective"],
        "speaking_style": agent_info["speaking_style"],
        "main_topic": state["main_topic"],
        "agent_position": agent_position,
        "other_participants": other_participants,
        "rag_context": rag_context,
    })
    
    # 清理并格式化响应
    response = response.strip()
    if not response.startswith(agent_info["name"]):
        response = f"{agent_info['name']}: {response}"
    
    print(f"🗣️ 开辩 {agent_info['name']}: {response}")
    
    # 更新状态
    new_total_messages = state.get("total_messages", 0) + 1
    new_stage_progress = state.get("stage_progress", 0) + 1
    
    # 保存开辩陈述
    opening_statements = state.get("opening_statements", {}).copy()
    opening_statements[agent_key] = response
    
    update_data = {
        "messages": [AIMessage(content=response)],
        "total_messages": new_total_messages,
        "stage_progress": new_stage_progress,
        "opening_statements": opening_statements,
    }
    
    # 更新缓存状态
    agent_paper_cache = state.get("agent_paper_cache", {})
    first_round_rag_completed = state.get("first_round_rag_completed", [])
    if agent_key in first_round_rag_completed:
        update_data["agent_paper_cache"] = agent_paper_cache
        update_data["first_round_rag_completed"] = first_round_rag_completed
    
    return update_data


def _generate_question(state: MultiAgentDebateState, agent_key: str) -> Dict[str, Any]:
    """生成提问"""
    agent_info = AVAILABLE_ROLES[agent_key]
    chat_template = create_questioning_chat_template()
    pipe = chat_template | deepseek | StrOutputParser()
    
    # 选择提问目标
    questions_asked = state.get("questions_asked", [])
    questioner, target = select_next_questioner_and_target(state["active_agents"], questions_asked)
    
    if agent_key != questioner:
        # 这种情况不应该发生，但提供一个安全退路
        error_msg = f"{agent_info['name']}: 当前不是我的提问时间。"
        return {
            "messages": [AIMessage(content=error_msg)],
            "total_messages": state.get("total_messages", 0) + 1,
            "stage_progress": state.get("stage_progress", 0) + 1,
        }
    
    target_name = AVAILABLE_ROLES[target]["name"]
    target_role = AVAILABLE_ROLES[target]["role"]
    
    # 格式化开辩陈述
    opening_statements_text = format_opening_statements(state.get("opening_statements", {}), state["active_agents"])
    
    # 获取其他参与者信息
    other_participants = get_other_participants(state["active_agents"], agent_key)
    
    # 获取联网搜索上下文
    rag_context = get_rag_context_for_agent(agent_key, state["main_topic"], state)
    
    # 构建提问任务描述
    task_description = f"""
现在是提问回答阶段，你有机会向 {target_name}({target_role}) 提出一个问题。

请基于：
1. 你作为{agent_info['role']}的专业角度
2. {target_name}在开辩阶段的陈述
3. 你希望深入了解或质疑的观点

向{target_name}提出一个具有挑战性和建设性的问题。
"""
    
    specific_instructions = f"问题要针对{target_name}的专业领域和观点，体现你的专业特色"
    
    # 调用模型生成提问
    response = pipe.invoke({
        "role": agent_info["role"],
        "name": agent_info["name"],
        "bio": agent_info["bio"],
        "main_topic": state["main_topic"],
        "other_participants": other_participants,
        "opening_statements": opening_statements_text,
        "rag_context": rag_context,
        "task_description": task_description,
        "specific_instructions": specific_instructions,
        "action_type": f"向{target_name}提问",
    })
    
    # 清理并格式化响应
    response = response.strip()
    if not response.startswith(agent_info["name"]):
        response = f"{agent_info['name']}: {response}"
    
    print(f"❓ 提问 {agent_info['name']} → {target_name}: {response}")
    
    # 更新状态
    new_total_messages = state.get("total_messages", 0) + 1
    new_stage_progress = state.get("stage_progress", 0) + 1
    
    # 记录问题
    new_questions = questions_asked.copy()
    new_questions.append({
        "questioner": agent_key,
        "target": target,
        "question": response,
        "answer": ""  # 待填入
    })
    
    update_data = {
        "messages": [AIMessage(content=response)],
        "total_messages": new_total_messages,
        "stage_progress": new_stage_progress,
        "questions_asked": new_questions,
        "current_questioner": agent_key,
        "current_target": target,
        "waiting_for_answer": True,
    }
    
    return update_data


def _generate_answer(state: MultiAgentDebateState, agent_key: str) -> Dict[str, Any]:
    """生成回答"""
    agent_info = AVAILABLE_ROLES[agent_key]
    chat_template = create_questioning_chat_template()
    pipe = chat_template | deepseek | StrOutputParser()
    
    # 获取最新的问题
    questions_asked = state.get("questions_asked", [])
    if not questions_asked:
        error_msg = f"{agent_info['name']}: 没有找到需要回答的问题。"
        return {
            "messages": [AIMessage(content=error_msg)],
            "total_messages": state.get("total_messages", 0) + 1,
            "stage_progress": state.get("stage_progress", 0) + 1,
        }
    
    latest_question = questions_asked[-1]
    questioner_name = AVAILABLE_ROLES[latest_question["questioner"]]["name"]
    question_content = latest_question["question"]
    
    # 格式化开辩陈述
    opening_statements_text = format_opening_statements(state.get("opening_statements", {}), state["active_agents"])
    
    # 获取其他参与者信息
    other_participants = get_other_participants(state["active_agents"], agent_key)
    
    # 获取联网搜索上下文
    rag_context = get_rag_context_for_agent(agent_key, state["main_topic"], state)
    
    # 构建回答任务描述
    task_description = f"""
{questioner_name}向你提出了以下问题：
"{question_content}"

请基于：
1. 你作为{agent_info['role']}的专业知识
2. 你在开辩阶段的立场
3. 联网搜索获得的最新资料

对这个问题进行专业、详实的回答。
"""
    
    specific_instructions = "回答要直接针对问题，体现你的专业观点，既要回应质疑也要坚持立场"
    
    # 调用模型生成回答
    response = pipe.invoke({
        "role": agent_info["role"],
        "name": agent_info["name"],
        "bio": agent_info["bio"],
        "main_topic": state["main_topic"],
        "other_participants": other_participants,
        "opening_statements": opening_statements_text,
        "rag_context": rag_context,
        "task_description": task_description,
        "specific_instructions": specific_instructions,
        "action_type": f"回答{questioner_name}的问题",
    })
    
    # 清理并格式化响应
    response = response.strip()
    if not response.startswith(agent_info["name"]):
        response = f"{agent_info['name']}: {response}"
    
    print(f"💬 回答 {agent_info['name']}: {response}")
    
    # 更新状态
    new_total_messages = state.get("total_messages", 0) + 1
    new_stage_progress = state.get("stage_progress", 0) + 1
    
    # 更新问题记录，添加回答
    updated_questions = questions_asked.copy()
    updated_questions[-1]["answer"] = response
    
    update_data = {
        "messages": [AIMessage(content=response)],
        "total_messages": new_total_messages,
        "stage_progress": new_stage_progress,
        "questions_asked": updated_questions,
        "waiting_for_answer": False,
        "current_questioner": "",
        "current_target": "",
    }
    
    return update_data


def _generate_free_debate_response(state: MultiAgentDebateState, agent_key: str) -> Dict[str, Any]:
    """生成自由辩论回复"""
    agent_info = AVAILABLE_ROLES[agent_key]
    chat_template = create_free_debate_chat_template()
    pipe = chat_template | deepseek | StrOutputParser()
    
    # 计算轮次和位置信息
    stage_progress = state.get("stage_progress", 0)
    active_agents_count = len(state["active_agents"])
    current_round = (stage_progress // active_agents_count) + 1
    agent_position_in_round = (stage_progress % active_agents_count) + 1
    
    # 格式化对话历史
    history = format_agent_history(state["messages"], state["active_agents"], agent_key, current_round)
    
    # 获取其他参与者信息
    other_participants = get_other_participants(state["active_agents"], agent_key)
    
    # 获取联网搜索上下文
    rag_context = get_rag_context_for_agent(agent_key, state["main_topic"], state)
    
    # 格式化前期内容摘要
    opening_summary = format_opening_statements(state.get("opening_statements", {}), state["active_agents"])
    qa_summary = format_qa_summary(state.get("questions_asked", []))
    
    # 调用模型生成回复
    response = pipe.invoke({
        "role": agent_info["role"],
        "name": agent_info["name"],
        "bio": agent_info["bio"],
        "focus": agent_info["focus"],
        "perspective": agent_info["perspective"],
        "speaking_style": agent_info["speaking_style"],
        "main_topic": state["main_topic"],
        "current_round": current_round,
        "max_rounds": state.get("max_rounds", 3),
        "agent_position": agent_position_in_round,
        "other_participants": other_participants,
        "rag_context": rag_context,
        "history": history,
        "opening_summary": opening_summary,
        "qa_summary": qa_summary,
    })
    
    # 清理并格式化响应
    response = response.strip()
    if not response.startswith(agent_info["name"]):
        response = f"{agent_info['name']}: {response}"
    
    print(f"🗣️ 自由辩论第{current_round}轮 {agent_info['name']}: {response}")
    
    # 更新状态
    new_total_messages = state.get("total_messages", 0) + 1
    new_stage_progress = state.get("stage_progress", 0) + 1
    
    update_data = {
        "messages": [AIMessage(content=response)],
        "total_messages": new_total_messages,
        "stage_progress": new_stage_progress,
    }
    
    return update_data


def _generate_closing_statement(state: MultiAgentDebateState, agent_key: str) -> Dict[str, Any]:
    """生成结辩陈述"""
    agent_info = AVAILABLE_ROLES[agent_key]
    chat_template = create_closing_chat_template()
    pipe = chat_template | deepseek | StrOutputParser()
    
    # 获取联网搜索上下文
    rag_context = get_rag_context_for_agent(agent_key, state["main_topic"], state)
    
    # 构建各种摘要
    your_position_summary = state.get("opening_statements", {}).get(agent_key, "未找到开辩陈述")
    
    # 其他专家的观点摘要
    others_positions = []
    for other_agent in state["active_agents"]:
        if other_agent != agent_key:
            other_name = AVAILABLE_ROLES[other_agent]["name"]
            other_statement = state.get("opening_statements", {}).get(other_agent, "")
            if other_statement:
                clean_statement = other_statement.replace(f"{other_name}:", "").strip()
                others_positions.append(f"{other_name}: {clean_statement[:100]}...")
    
    others_positions_summary = "\n".join(others_positions)
    
    # 关键争议点（简化处理）
    key_controversies = "在辩论中出现的主要分歧包括：技术发展速度与社会适应能力的平衡、经济效益与社会公平的权衡、以及监管政策的必要性和边界等问题。"
    
    # 调用模型生成结辩陈词
    response = pipe.invoke({
        "role": agent_info["role"],
        "name": agent_info["name"],
        "bio": agent_info["bio"],
        "main_topic": state["main_topic"],
        "your_position_summary": your_position_summary,
        "others_positions_summary": others_positions_summary,
        "key_controversies": key_controversies,
        "rag_context": rag_context,
    })
    
    # 清理并格式化响应
    response = response.strip()
    if not response.startswith(agent_info["name"]):
        response = f"{agent_info['name']}: {response}"
    
    print(f"🏁 结辩 {agent_info['name']}: {response}")
    
    # 更新状态
    new_total_messages = state.get("total_messages", 0) + 1
    new_stage_progress = state.get("stage_progress", 0) + 1
    
    # 保存结辩陈述
    closing_statements = state.get("closing_statements", {}).copy()
    closing_statements[agent_key] = response
    
    update_data = {
        "messages": [AIMessage(content=response)],
        "total_messages": new_total_messages,
        "stage_progress": new_stage_progress,
        "closing_statements": closing_statements,
    }
    
    return update_data


def create_agent_node_function(agent_key: str):
    """为指定Agent创建节点函数（四阶段版）"""
    def agent_node(state: MultiAgentDebateState) -> Command:
        try:
            current_stage = state.get("current_stage", "opening")
            stage_progress = state.get("stage_progress", 0)
            active_agents = state.get("active_agents", [])
            max_rounds = state.get("max_rounds", 3)
            
            # 确定下一个应该发言的节点
            next_node = determine_next_node(state)
            
            # 处理阶段转换
            if next_node in ["questioning", "free_debate", "closing"]:
                return handle_stage_transition(state, next_node, agent_key)
            
            # 如果是结束节点
            if next_node == END:
                print("🏁 辩论结束")
                return Command(update={"messages": []}, goto=END)
            
            # 如果当前不是该agent的发言时间，跳转
            if next_node != agent_key:
                print(f"🔄 跳转到正确的发言者：{next_node}")
                return Command(update={"messages": []}, goto=next_node)
            
            # 生成回复
            try:
                update_data = _generate_agent_response(state, agent_key)
                
                if not update_data or "messages" not in update_data:
                    print(f"❌ {agent_key} 生成的回复数据无效")
                    update_data = {
                        "messages": [AIMessage(content=f"{AVAILABLE_ROLES[agent_key]['name']}: 抱歉，我现在无法发言。")],
                        "total_messages": state.get("total_messages", 0) + 1,
                        "stage_progress": state.get("stage_progress", 0) + 1,
                    }
                
                # 确定下一个节点（基于更新后的状态）
                updated_state = {**state, **update_data}
                next_node = determine_next_node(updated_state)
                
                print(f"📊 当前阶段：{updated_state.get('current_stage')}，进度：{updated_state.get('stage_progress')}，下一个：{next_node}")
                
                return Command(update=update_data, goto=next_node)
                
            except Exception as e:
                print(f"❌ 专家 {agent_key} 发言失败: {e}")
                error_update = {
                    "messages": [AIMessage(content=f"{AVAILABLE_ROLES[agent_key]['name']}: 抱歉，技术问题导致无法发言。")],
                    "total_messages": state.get("total_messages", 0) + 1,
                    "stage_progress": state.get("stage_progress", 0) + 1,
                }
                return Command(update=error_update, goto=END)
        
        except Exception as e:
            print(f"❌ 专家节点 {agent_key} 处理失败: {e}")
            safe_update = {
                "messages": [AIMessage(content=f"系统错误：{agent_key} 无法处理")],
                "total_messages": state.get("total_messages", 0) + 1,
                "stage_progress": state.get("stage_progress", 0) + 1,
            }
            return Command(update=safe_update, goto=END)
    
    return agent_node


def handle_stage_transition(state: MultiAgentDebateState, target_stage: str, current_agent: str) -> Command:
    """处理阶段转换"""
    try:
        stage_messages = {
            "questioning": "📝 现在进入提问回答阶段，每位专家将有机会向其他专家提问。",
            "free_debate": "🎯 现在进入自由辩论阶段，各位专家可以就争议观点展开深入讨论。",
            "closing": "🏁 现在进入结辩综述阶段，各位专家请发表总结陈词。"
        }
        
        print(f"🔄 {current_agent} 触发阶段转换：{state.get('current_stage')} -> {target_stage}")
        
        update_data = {
            "current_stage": target_stage,
            "stage_progress": 0,
            "messages": [AIMessage(content=stage_messages.get(target_stage, f"转换到{target_stage}阶段"))]
        }
        
        # 特殊处理提问阶段的初始化
        if target_stage == "questioning":
            update_data.update({
                "questions_asked": [],
                "current_questioner": "",
                "current_target": "",
                "waiting_for_answer": False
            })
        
        # 确定转换后的第一个发言者
        next_node = determine_next_node({**state, **update_data})
        
        return Command(update=update_data, goto=next_node)
        
    except Exception as e:
        print(f"❌ 阶段转换失败: {e}")
        return Command(update={"messages": []}, goto=END)


def create_multi_agent_graph(active_agents: List[str], rag_enabled: bool = True) -> StateGraph:
    """创建多角色辩论图（四阶段版）"""
    if len(active_agents) < 3:
        raise ValueError("至少需要3个Agent参与辩论")
    
    if len(active_agents) > 6:
        raise ValueError("最多支持6个Agent参与辩论")
    
    # 验证所有Agent都存在
    for agent_key in active_agents:
        if agent_key not in AVAILABLE_ROLES:
            raise ValueError(f"未知的Agent: {agent_key}")
    
    builder = StateGraph(MultiAgentDebateState)
    for agent_key in active_agents:
        agent_function = create_agent_node_function(agent_key)
        builder.add_node(agent_key, agent_function)

    # 关键：注册阶段转换节点
    builder.add_node("questioning", lambda state: handle_stage_transition(state, "questioning", "system"))
    builder.add_node("free_debate", lambda state: handle_stage_transition(state, "free_debate", "system"))
    builder.add_node("closing", lambda state: handle_stage_transition(state, "closing", "system"))

    first_agent = active_agents[0]
    builder.add_edge(START, first_agent)
    return builder.compile()


def test_four_stage_multi_agent_debate(topic: str = "人工智能对教育的影响", 
                                     rounds: int = 3, 
                                     agents: List[str] = None,
                                     enable_rag: bool = True,
                                     max_refs_per_agent: int = 3):
    """测试四阶段多角色辩论功能"""
    if agents is None:
        agents = ["tech_expert", "sociologist", "ethicist"]
    
    print(f"🎯 开始测试四阶段多角色辩论: {topic}")
    print(f"👥 参与者: {[AVAILABLE_ROLES[k]['name'] for k in agents]}")
    print(f"📊 自由辩论轮数: {rounds}")
    print(f"🌐 联网搜索: {'启用' if enable_rag else '禁用'}")
    print(f"🎭 四阶段流程: 开辩综述 → 提问回答 → 自由辩论 → 结辩综述")
    print("=" * 70)
    
    try:
        test_graph = create_multi_agent_graph(agents, rag_enabled=enable_rag)
        
        inputs = {
            "main_topic": topic,
            "messages": [],
            "current_stage": "opening",
            "stage_progress": 0,
            "max_rounds": rounds,
            "active_agents": agents,
            "total_messages": 0,
            "rag_enabled": enable_rag,
            "rag_sources": ["web_search"],
            "collected_references": [],
            "max_refs_per_agent": max_refs_per_agent,
            "max_results_per_source": 2,
            "agent_paper_cache": {},
            "first_round_rag_completed": [],
            # 四阶段相关字段
            "questions_asked": [],
            "current_questioner": "",
            "current_target": "",
            "waiting_for_answer": False,
            "opening_statements": {},
            "closing_statements": {},
            # 简化的追踪字段
            "agent_positions": {},
            "key_points_raised": [],
            "controversial_points": []
        }
        
        for i, output in enumerate(test_graph.stream(inputs, stream_mode="updates"), 1):
            print(f"消息 {i}: {output}")
            
        print("=" * 70)
        print("✅ 四阶段多角色辩论测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")


# 工具函数：预热联网搜索系统
def warmup_rag_system(test_topic: str = "人工智能"):
    """预热联网搜索系统，测试API连接"""
    if rag_module:
        print("🔥 预热联网搜索系统...")
        try:
            test_results = rag_module.search_academic_sources(test_topic, max_results_per_source=1)
            if test_results:
                print("✅ 联网搜索系统预热完成，API连接正常")
            else:
                print("⚠️ 联网搜索系统预热完成，但未搜索到测试结果")
        except Exception as e:
            print(f"⚠️ 联网搜索系统预热失败: {e}")


# 主程序入口
if __name__ == "__main__":
    # 检查环境变量
    missing_keys = []
    if not os.getenv("DEEPSEEK_API_KEY"):
        missing_keys.append("DEEPSEEK_API_KEY")
    if not os.getenv("KIMI_API_KEY"):
        missing_keys.append("KIMI_API_KEY")
    
    if missing_keys:
        print(f"❌ 警告: {', '.join(missing_keys)} 环境变量未设置")
        print("请设置以下环境变量：")
        for key in missing_keys:
            print(f"export {key}=your_api_key")
    else:
        print("✅ 环境变量配置正确")
        
        # 预热联网搜索系统
        warmup_rag_system()
        
        # 测试四阶段辩论
        test_four_stage_multi_agent_debate(
            topic="ChatGPT对教育的影响",
            rounds=3,
            agents=["tech_expert", "sociologist", "ethicist"],
            enable_rag=True,
            max_refs_per_agent=3
        )