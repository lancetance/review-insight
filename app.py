# -*- coding: utf-8 -*-
# AI 用户洞察助手 —— 本地 Streamlit 应用(DeepSeek 版)
# 功能:三种方式输入评论(粘贴文本/Excel/CSV) -> 按"商品评价类"或"视频社区讨论类"
#       两套不同框架调用 DeepSeek 分析 -> 显示报告 -> 支持下载

import streamlit as st          # 网页框架
import pandas as pd             # 读取 Excel / CSV 表格
from openai import OpenAI       # DeepSeek 用 OpenAI 兼容接口,所以用 openai 这个库
import os

# ========== 页面基本设置 ==========
st.set_page_config(page_title="AI 用户洞察助手", page_icon="📊", layout="wide")
st.title("📊 AI 用户洞察助手")
st.caption("粘贴文本或上传商品评价 / 视频评论,自动生成主题归类、情绪分析与洞察报告")

# ========== 读取 API Key ==========
# Key 不写在代码里:优先从 st.secrets 读取(部署到 Streamlit Community Cloud 时使用),
# 本地没有 secrets.toml 时会抛异常,这里捕获后回退到环境变量 DEEPSEEK_API_KEY。
try:
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

if not API_KEY:
    st.error("未检测到 API Key。请先设置环境变量 DEEPSEEK_API_KEY 再运行。")
    st.stop()

# 注意 base_url:这一行是关键,它告诉程序去 DeepSeek 的服务器,而不是 OpenAI 的
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# ========== 访问密码保护 ==========
# 正确密码从 st.secrets 的 APP_PASSWORD 读取;本地没有配置该项时,跳过密码保护方便调试。
try:
    APP_PASSWORD = st.secrets["APP_PASSWORD"]
except Exception:
    APP_PASSWORD = None

if APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        password_input = st.text_input("请输入访问密码", type="password")
        if password_input == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            if password_input:
                st.error("密码错误,请重新输入。")
            else:
                st.info("请输入访问密码以继续。")
            st.stop()

# ========== 侧边栏:控制发送给模型的文本长度上限 ==========
with st.sidebar:
    st.header("设置")
    max_chars = st.slider(
        "发送分析的文本长度上限(字符数)",
        min_value=5000,
        max_value=50000,
        value=20000,
        step=1000,
    )
    st.caption("超出上限的部分会被截断后再发送,避免一次性发送过多内容导致报错或浪费额度。")

# ========== 通用的"噪音过滤"说明,两套框架都会用到 ==========
NOISE_FILTER_INSTRUCTION = """原始数据中可能夹杂用户名、日期、点赞数等信息,请这样理解和使用它们:
- 点赞数:代表该评论的认同度/影响力,在做"严重度排序""高赞引爆点"等分析时必须重点利用;
- 日期:可用于观察问题或讨论的时间趋势;
- 用户名:报告中一般不需要直接体现,但可以帮助你区分不同评论之间的边界。

真正需要过滤、不计入分析的"噪音"只有:
- "回复"、"共X条回复"等界面交互文字、广告或自我推销(如"求点赞我写了几首诗")
- 纯 @某人、纯艾特、单个 emoji、纯灌水的口水话(如"绷住""哈哈""沙发""打卡")
- 评论文本中混入的 markdown 链接、图片地址等格式垃圾(请忽略这些格式,只看其中的文字内容)

请先在心里识别并过滤掉以上噪音,再基于剩下"有信息量"的评论进行下面的分析。"""

# ========== 通用的输出格式要求,两套框架都会用到 ==========
OUTPUT_FORMAT_INSTRUCTION = """请直接输出报告正文,不要以"作为一名资深的XX分析师"之类的开场白或自我介绍开头。"""


# ========== 分析框架一:商品评价类 ==========
def build_product_prompt(reviews_text):
    """构造"商品评价类"分析的 prompt。"""
    prompt = f"""你是一位消费者研究分析师。下面是某商品的一批用户评论,可能同时包含好评与差评,
也可能包含用户名、日期、点赞数等额外信息。

{NOISE_FILTER_INSTRUCTION}

{OUTPUT_FORMAT_INSTRUCTION}

请用中文完成以下分析,并用清晰的小标题分段输出。注意:数据中可能同时存在好评和差评,
两者都要分析,不要只关注差评;各部分内容应聚焦各自的重点,不要互相重复。

① 整体概览
   给出好评与差评的大致比例、综合评价倾向,并用一句话给出总体结论。

② 用户称赞的优点
   从好评中归纳用户反复称赞的方面(数据驱动,不要预设固定维度),
   每个优点给出被提及的大致次数和 1~2 个典型例子原文。

③ 用户抱怨的问题
   从差评中归纳主要问题(数据驱动,不要预设固定维度),按严重程度排序,
   排序依据为:被提及的频率 + 点赞数/"觉得有用"数(如果数据中有这些信息)。
   明确指出头号问题是什么,并指出哪些是多人反复提及的系统性问题(更可能是批次/设计缺陷而非个例)。

④ 深层洞察
   挖掘用户负面情绪的根源(例如被欺骗感、预期落差、信任崩塌等),
   并指出整体上最该重视的核心点是什么。如果数据中带有日期信息,可留意时间趋势
   (例如品质是否在某个时间点之后明显下降)。

⑤ 建议与购买参考
   给商家 2~3 条具体、可落地的改进建议;再给消费者一句购买参考(是否值得购买、需要注意什么)。

评论数据如下(每行一条,行内可能夹杂用户名、日期、点赞数等信息):
{reviews_text}
"""
    return prompt


# ========== 分析框架二:视频社区讨论类 ==========
def build_video_prompt(reviews_text, video_title, video_intro):
    """构造"视频社区讨论类"分析的 prompt,会用到视频标题、简介/标签。"""
    intro_part = video_intro.strip() if video_intro and video_intro.strip() else "(未填写)"

    prompt = f"""你是一位社区内容与用户研究分析师。下面是某个视频下方的一批评论,可能包含用户名、日期、点赞数等额外信息。

视频标题:{video_title}
视频简介/标签:{intro_part}

{NOISE_FILTER_INSTRUCTION}

{OUTPUT_FORMAT_INSTRUCTION}

请按以下步骤完成分析,用中文输出,并用清晰的小标题分段:

第一步:判断视频类型与调性
结合视频标题、简介/标签,先判断这个视频大致属于什么类型(例如搞笑、访谈、游戏、知识科普、
生活记录、情感等),以及大致的调性(轻松/严肃/温情等)。

第二步:在以下五个固定维度上分析评论区,并根据上一步判断出的视频类型,
自动调整每个维度的分析侧重(例如搞笑视频重点挖"高频梗/二创",访谈类重点挖"观点讨论/情感共鸣",
游戏类重点看"吐槽/争议"):

1. 评论区主要构成类型
   识别评论主要由哪几类构成、各占大致比例。常见类型包括:玩梗/二创、情感共鸣/个人故事、
   观点讨论、对创作者的反馈/期待、捧场灌水等。

2. 高赞引爆点
   不要逐条复述高赞评论的原文内容,而是分析"为什么这几条评论能获得最高的点赞数"——
   结合评论区整体氛围,归纳这些高赞内容的共性规律和引爆原因(例如说出了大多数人的心声、
   提供了独特视角、玩出了高级的梗等)。(请充分利用数据中的点赞数信息,如果有的话)

3. 评论区特色表达/社区文化
   如果评论区有明显的"梗"(玩梗、黑话、接龙等),分析这些梗及其体现的社区文化;
   如果没有特别突出的"梗文化",则改为分析这个评论区独特的表达方式或讨论风格
   (例如惯用的语气、句式、互动习惯等),不要为了凑梗而牵强附会。

4. 整体情感基调
   评论区的集体情绪氛围是什么?(例如亢奋玩乐、温暖治愈、引发思考、吐槽抱怨等)

5. 给创作者的洞察
   观众对内容的反馈、期待、对创作者的态度;可以作为后续选题或创作方向参考的洞察。

评论数据如下(每行一条,行内可能夹杂用户名、日期、点赞数等信息):
{reviews_text}
"""
    return prompt


# ========== 调用 DeepSeek 的函数 ==========
def analyze_reviews(prompt):
    """把构造好的 prompt 发给 DeepSeek,返回分析结果文字。"""
    response = client.chat.completions.create(
        model="deepseek-chat",       # DeepSeek 的对话模型
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ========== 第一步:选择评论类型 ==========
st.subheader("① 选择评论类型")
comment_type = st.radio(
    "这批评论属于哪一类?不同类型会使用不同的分析框架。",
    ["商品评价类", "视频社区讨论类"],
    horizontal=True,
)

# 如果是视频类,额外要求填写标题,并可选填简介/标签
video_title, video_intro = "", ""
if comment_type == "视频社区讨论类":
    video_title = st.text_input("视频标题(必填)")
    video_intro = st.text_area("视频简介/标签(选填)", height=80)

st.divider()

# ========== 第二步:选择输入方式 ==========
st.subheader("② 选择输入方式")
input_method = st.radio(
    "评论数据怎么提供?",
    ["粘贴文本", "上传 Excel", "上传 CSV"],
    horizontal=True,
)

raw_text = ""  # 最终发送给模型分析的文本

if input_method == "粘贴文本":
    # 用 session_state 保存文本框内容,方便"一键清除"按钮重置它
    if "pasted_text" not in st.session_state:
        st.session_state.pasted_text = ""

    def clear_pasted_text():
        st.session_state.pasted_text = ""

    label_col, button_col = st.columns([5, 1])
    with label_col:
        st.write("把从网页复制的评论粘贴在这里(包含用户名、日期、点赞数等都没关系,模型会自行识别)")
    with button_col:
        st.button("一键清除", on_click=clear_pasted_text)

    pasted_text = st.text_area(
        "粘贴评论",
        height=250,
        key="pasted_text",
        label_visibility="collapsed",
    )
    # 直接把整段原始文本交给模型分析,不在本地切分/过滤
    raw_text = pasted_text

else:
    file_type = "xlsx" if input_method == "上传 Excel" else "csv"
    uploaded = st.file_uploader(
        f"上传评论文件({file_type.upper()} 格式)",
        type=[file_type],
    )

    if uploaded is not None:
        # 根据输入方式选择对应的 pandas 读取函数
        if input_method == "上传 Excel":
            df = pd.read_excel(uploaded)  # 需要 openpyxl 库支持
        else:
            df = pd.read_csv(uploaded)

        st.write(f"共读取到 {len(df)} 行数据")
        st.dataframe(df.head(10))

        # 让用户选择哪一列是评论文本(因为不同表格列名可能不同)
        text_column = st.selectbox("请选择包含评论文本的列", df.columns)
        reviews = df[text_column].dropna().astype(str).tolist()
        raw_text = "\n".join(f"- {r}" for r in reviews)

st.divider()

# ========== 第三步:开始分析 ==========
st.subheader("③ 开始分析")

# 视频类型必须填写标题才能分析
ready = bool(raw_text.strip()) and (comment_type != "视频社区讨论类" or video_title.strip())

if not raw_text.strip():
    st.info("👆 请先粘贴文本或上传文件,确保至少包含一条评论内容。")
elif comment_type == "视频社区讨论类" and not video_title.strip():
    st.warning("视频社区讨论类需要填写视频标题,请在上方补充。")

if st.button("开始分析", type="primary", disabled=not ready):
    # 按字符数截断,避免一次性发送过多内容导致报错或浪费额度
    reviews_text = raw_text
    if len(reviews_text) > max_chars:
        reviews_text = reviews_text[:max_chars]
        st.info(f"文本长度超过 {max_chars} 字符,已截断后再发送分析。")

    # 根据评论类型构造不同的 prompt
    if comment_type == "商品评价类":
        prompt = build_product_prompt(reviews_text)
    else:
        prompt = build_video_prompt(reviews_text, video_title, video_intro)

    with st.spinner("正在分析,请稍候..."):
        try:
            result = analyze_reviews(prompt)
            st.success("分析完成!")
            st.subheader("分析报告")
            st.markdown(result)

            # 提供下载报告的按钮
            st.download_button(
                "下载报告(txt)",
                data=result,
                file_name="洞察报告.txt",
            )
        except Exception as e:
            st.error(f"分析出错:{e}")
