# -*- coding: utf-8 -*-

import os
import re
import json
import base64
from pathlib import Path
from datetime import datetime

import streamlit as st
import pymupdf
from docx import Document
from openai import OpenAI


# ============================================================
# 页面设置
# ============================================================

st.set_page_config(
    page_title="AI 简历筛选 Agent",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI 简历筛选 Agent")
st.caption("JD解析 → 简历解析 → 证据提取 → Rubric评分 → 证据封顶 → 加权排序")


# ============================================================
# 评分标准
# ============================================================

ALLOWED_SCORES = [0, 2, 4, 6, 8, 10]

EVIDENCE_MAX_SCORE = {
    "A": 10,
    "B": 8,
    "C": 6,
    "D": 0
}


RUBRIC = {
    "学历匹配": {
        10: "学历满足要求，且专业高度直接相关。",
        8: "学历满足要求，专业相关，但不是最直接匹配。",
        6: "学历满足要求，专业存在一定相关性。",
        4: "学历满足要求，但专业基本无关。",
        2: "学历不满足要求，但专业高度相关，存在一定补偿性。",
        0: "学历明显不满足要求，且专业无法弥补。"
    },

    "工作经验": {
        10: "相关正式工作经验达到或超过JD要求，且相关性高。",
        8: "相关正式工作经验达到要求的80%-99%，且相关性较高；或存在轻微相关性不足。",
        6: "相关正式工作经验达到要求的50%-79%，且存在明确相关经验。",
        4: "相关正式工作经验达到要求的25%-49%，但经验明显不足。",
        2: "相关正式工作经验达到要求的1%-24%，或主要只有边缘相关经验。",
        0: "没有有效的相关正式工作经验。"
    },

    "岗位相关经验": {
        10: "核心工作内容高度匹配，有充分直接证据。",
        8: "大部分核心工作内容高度匹配，仅存在小幅差异。",
        6: "存在明显相关经验，但只覆盖部分核心内容。",
        4: "只有外围或部分工作内容相关。",
        2: "相关性很弱，仅存在间接关联。",
        0: "没有实质相关经验。"
    },

    "核心技能": {
        10: "90%以上核心技能有直接使用证据。",
        8: "75%-89%核心技能有直接使用证据。",
        6: "50%-74%核心技能有直接或部分使用证据。",
        4: "25%-49%核心技能有相关证据。",
        2: "仅有少量间接证据。",
        0: "没有有效技能证据。"
    },

    "项目经验": {
        10: "有高度对应项目，并有明确行动、职责或结果证据。",
        8: "项目高度相关，但存在小幅覆盖缺口。",
        6: "项目明显相关，但只覆盖部分要求。",
        4: "只有外围相关项目。",
        2: "项目相关性弱，或描述非常模糊。",
        0: "没有相关项目证据。"
    },

    "职责匹配": {
        10: "90%以上核心职责有直接证据，且没有核心职责缺失。",
        8: "75%-89%的职责有直接证据。",
        6: "50%-74%的职责有明确匹配。",
        4: "25%-49%的职责有匹配。",
        2: "存在少量弱相关职责证据。",
        0: "没有对应职责证据。"
    }
}


WEIGHTS_DEFAULT = {
    "学历匹配": 10,
    "工作经验": 15,
    "岗位相关经验": 25,
    "核心技能": 20,
    "项目经验": 15,
    "职责匹配": 15
}


# ============================================================
# AI Provider
# ============================================================

PROVIDERS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "vision": False
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "vision": True
    },
    "智谱 GLM": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "default_model": "glm-4v-flash",
        "vision": True
    },
    "通义千问": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-vl-max",
        "vision": True
    }
}


# ============================================================
# 基础工具
# ============================================================

def safe_json_load(text):
    """
    尽可能从模型输出中提取 JSON。
    """
    if not text:
        raise ValueError("模型没有返回内容")

    text = text.strip()

    # 去掉 markdown code fence
    text = re.sub(r"^```json\s*", "", text, flags=re.I)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    # 尝试提取第一个 JSON 对象
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("无法解析模型返回的 JSON")


def normalize_score(score):
    """
    强制所有分数只能是：
    0 / 2 / 4 / 6 / 8 / 10
    """

    try:
        score = float(score)
    except Exception:
        return 0

    # 如果模型返回了非法分数：
    # 例如 7、9，则映射到最近的合法分数
    return min(
        ALLOWED_SCORES,
        key=lambda x: abs(x - score)
    )


def apply_evidence_cap(score, evidence_grade):
    """
    根据证据等级进行二次封顶。

    A -> 最高10
    B -> 最高8
    C -> 最高6
    D -> 0
    """

    score = normalize_score(score)

    grade = str(evidence_grade or "D").upper().strip()

    if grade not in EVIDENCE_MAX_SCORE:
        grade = "D"

    max_score = EVIDENCE_MAX_SCORE[grade]

    return min(score, max_score)


def clean_text(text):
    if not text:
        return ""

    return str(text).replace("\r\n", "\n").strip()


# ============================================================
# 文件读取
# ============================================================

TEXT_EXTENSIONS = {
    ".txt",
    ".md"
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS
    | IMAGE_EXTENSIONS
    | {".docx", ".pdf"}
)


def read_txt_file(file_path):
    path = Path(file_path)

    for encoding in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue

    raise ValueError(f"无法读取文本文件：{path.name}")


def read_docx(file_path):
    """
    同时读取 Word 段落和表格。
    """

    doc = Document(file_path)

    parts = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if text:
            parts.append(text)

    for table in doc.tables:
        for row in table.rows:
            row_text = []

            for cell in row.cells:
                text = cell.text.strip()

                if text:
                    row_text.append(text)

            if row_text:
                parts.append(" | ".join(row_text))

    return "\n".join(parts)


def read_pdf_text(file_path):
    """
    尝试读取 PDF 文本层。
    """

    doc = pymupdf.open(file_path)

    pages = []

    for page in doc:
        text = page.get_text("text")

        if text:
            pages.append(text)

    doc.close()

    return "\n".join(pages).strip()


def pdf_to_images(file_path, max_pages=8, scale=1.5):
    """
    将 PDF 前 max_pages 页转换为图片。
    """

    doc = pymupdf.open(file_path)

    images = []

    page_count = min(len(doc), max_pages)

    for i in range(page_count):
        page = doc.load_page(i)

        matrix = pymupdf.Matrix(scale, scale)

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        images.append(pix.tobytes("png"))

    doc.close()

    return images


def image_to_base64(image_bytes):
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    return encoded


# ============================================================
# AI Client
# ============================================================

def create_client(provider_name, api_key):

    config = PROVIDERS[provider_name]

    return OpenAI(
        api_key=api_key,
        base_url=config["base_url"]
    )


# ============================================================
# 普通文本 AI 请求
# ============================================================

def call_text_model(
    client,
    model,
    system_prompt,
    user_prompt
):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ============================================================
# Vision AI 请求
# ============================================================

def call_vision_model(
    client,
    model,
    system_prompt,
    user_prompt,
    image_bytes_list
):

    content = [
        {
            "type": "text",
            "text": user_prompt
        }
    ]

    for image_bytes in image_bytes_list:

        base64_data = image_to_base64(image_bytes)

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_data}"
                }
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ============================================================
# 文件内容统一入口
# ============================================================

def extract_file_content(
    file_path,
    client,
    model,
    provider_name
):

    path = Path(file_path)

    suffix = path.suffix.lower()

    # -----------------------------
    # 文本
    # -----------------------------

    if suffix in TEXT_EXTENSIONS:

        return {
            "type": "text",
            "text": read_txt_file(path),
            "images": []
        }

    # -----------------------------
    # DOCX
    # -----------------------------

    if suffix == ".docx":

        return {
            "type": "text",
            "text": read_docx(path),
            "images": []
        }

    # -----------------------------
    # 图片
    # -----------------------------

    if suffix in IMAGE_EXTENSIONS:

        if not PROVIDERS[provider_name]["vision"]:
            raise ValueError(
                f"{provider_name} 当前配置不支持图片视觉解析。"
            )

        image_bytes = path.read_bytes()

        return {
            "type": "vision",
            "text": "",
            "images": [image_bytes]
        }

    # -----------------------------
    # PDF
    # -----------------------------

    if suffix == ".pdf":

        text = read_pdf_text(path)

        # 文本层正常
        if len(text.strip()) >= 80:

            return {
                "type": "text",
                "text": text,
                "images": []
            }

        # 扫描 PDF
        if not PROVIDERS[provider_name]["vision"]:
            raise ValueError(
                f"{path.name} 看起来是扫描版 PDF，"
                f"需要支持视觉输入的模型。"
            )

        images = pdf_to_images(path)

        return {
            "type": "vision",
            "text": "",
            "images": images
        }

    # -----------------------------
    # DOC
    # -----------------------------

    if suffix == ".doc":

        raise ValueError(
            f"{path.name} 是旧版 .doc 文件。\n\n"
            f"请使用 Word 另存为 .docx 或 .pdf 后重新上传。"
        )

    raise ValueError(
        f"不支持的文件格式：{suffix}"
    )


# ============================================================
# JD 解析
# ============================================================

JD_SYSTEM_PROMPT = """
你是一名招聘JD结构化分析专家。

你的任务不是评价候选人，而是从JD中提取客观招聘要求。

只能使用JD明确出现的信息。

不要自行补充JD没有写出的要求。

输出必须是合法JSON，不要输出Markdown。
"""


def parse_jd(
    client,
    model,
    jd_text
):

    prompt = f"""
请分析下面的招聘JD。

提取：

1. 岗位名称
2. 学历要求
3. 专业要求
4. 工作经验要求
5. 工作经验要求对应的大致年限
6. 核心技能
7. 岗位职责
8. 核心职责
9. 项目经验要求

如果JD没有明确写某项，请填写空字符串或空数组。

工作经验年限：
如果JD写“2年以上”，required_experience_months填写24。
如果写“3年以上”，填写36。
如果没有明确年限，填写0。

请输出：

{{
    "岗位名称": "",
    "学历要求": "",
    "专业要求": [],
    "工作经验要求": "",
    "required_experience_months": 0,
    "技能要求": [],
    "岗位职责": [],
    "核心职责": [],
    "项目经验要求": []
}}

JD：

{jd_text}
"""

    result = call_text_model(
        client,
        model,
        JD_SYSTEM_PROMPT,
        prompt
    )

    return safe_json_load(result)


# ============================================================
# 简历解析
# ============================================================

RESUME_SYSTEM_PROMPT = """
你是一名简历信息结构化专家。

你的任务是从简历中提取事实证据。

严格禁止：

1. 编造简历中没有出现的经历。
2. 把“熟悉”自动理解成“实际负责”。
3. 把“了解”自动理解成“熟练”。
4. 把推测内容当成事实。
5. 把实习自动计算为正式工作经验。

所有信息必须能够在简历中找到依据。

输出必须是合法JSON，不要输出Markdown。
"""


def parse_resume(
    client,
    model,
    resume_text
):

    prompt = f"""
请结构化提取下面这份候选人简历。

输出：

{{
    "姓名": "",
    "学历": "",
    "专业": "",

    "工作经历": [
        {{
            "公司": "",
            "职位": "",
            "时间": "",
            "是否实习": false,
            "工作内容": []
        }}
    ],

    "项目经历": [
        {{
            "项目名称": "",
            "时间": "",
            "项目内容": [],
            "结果": []
        }}
    ],

    "技能": [],

    "其他信息": []
}}

注意：

- “2022.07-至今”必须原样保留。
- “2022.07-2024.06”必须原样保留。
- 如果是实习，是否实习必须为true。
- 不确定是否实习时，不要擅自判断为正式工作。
- 工作内容尽量拆成具体行为。
- 不要把JD中的要求补进候选人简历。

候选人简历：

{resume_text}
"""

    result = call_text_model(
        client,
        model,
        RESUME_SYSTEM_PROMPT,
        prompt
    )

    return safe_json_load(result)


# ============================================================
# Rubric Prompt
# ============================================================

def rubric_to_text():

    parts = []

    for dimension, rules in RUBRIC.items():

        parts.append(f"【{dimension}】")

        for score in ALLOWED_SCORES:
            parts.append(
                f"{score}分：{rules[score]}"
            )

        parts.append("")

    parts.append(
        """
【证据等级】

A：
简历中存在明确、直接、具体的证据。
最高可以得到10分。

B：
根据简历中的明确行为可以高度合理地判断存在该能力，
但缺乏完全直接的证明。
最高可以得到8分。

C：
只有间接相关证据。
最高可以得到6分。

D：
没有有效证据。
只能得到0分。

【绝对规则】

1. 分数只能是0、2、4、6、8、10。
2. 不允许输出其他分数。
3. D证据必须为0分。
4. B证据最高8分。
5. C证据最高6分。
6. A证据最高10分。
7. 不允许因为候选人的学历高、公司好、工作年限长而自动提高其他维度。
8. 工作经验和岗位相关经验必须分开判断。
9. 项目经验不能因为工作经历存在而自动获得高分。
10. 技能必须有简历证据，不允许因为职位名称而自动推断技能。
"""
    )

    return "\n".join(parts)


# ============================================================
# 候选人评分
# ============================================================

SCORING_SYSTEM_PROMPT = """
你是一名严格的AI简历评估员。

你的任务是：

JD要求
↓
候选人简历
↓
寻找直接证据
↓
按照Rubric评分

不是凭整体印象评分。

必须区分：

- 有证据
- 间接证据
- 没有证据

不要因为候选人整体看起来不错，就给所有维度高分。

所有评分必须能够被具体证据解释。

输出必须是合法JSON。
不要输出Markdown。
"""


def score_candidate(
    client,
    model,
    jd_data,
    resume_data
):

    rubric_text = rubric_to_text()

    prompt = f"""
下面是本系统固定使用的Rubric：

{rubric_text}

========================

【JD】

{json.dumps(jd_data, ensure_ascii=False, indent=2)}

========================

【候选人】

{json.dumps(resume_data, ensure_ascii=False, indent=2)}

========================

请按照Rubric评估候选人。

必须返回以下JSON结构：

{{
    "学历匹配": {{
        "score": 0,
        "evidence_grade": "A",
        "evidence": "",
        "reason": ""
    }},

    "工作经验": {{
        "score": 0,
        "evidence_grade": "A",
        "evidence": "",
        "reason": "",
        "relevant_work_records": [
            {{
                "公司": "",
                "职位": "",
                "时间": "",
                "是否实习": false,
                "相关性": "高/中/低"
            }}
        ]
    }},

    "岗位相关经验": {{
        "score": 0,
        "evidence_grade": "A",
        "evidence": "",
        "reason": "",
        "matched_duties": [],
        "unmatched_duties": []
    }},

    "核心技能": {{
        "score": 0,
        "evidence_grade": "A",
        "evidence": "",
        "reason": "",
        "matched_skills": [],
        "unmatched_skills": []
    }},

    "项目经验": {{
        "score": 0,
        "evidence_grade": "A",
        "evidence": "",
        "reason": "",
        "matched_projects": [],
        "unmatched_requirements": []
    }},

    "职责匹配": {{
        "score": 0,
        "evidence_grade": "A",
        "evidence": "",
        "reason": "",
        "duty_analysis": [
            {{
                "jd_duty": "",
                "match": "直接/部分/无",
                "evidence": ""
            }}
        ]
    }}
}}

========================

【评分要求】

### 1. 学历匹配

严格根据JD学历和专业要求判断。

不要因为候选人工作经验丰富而提高学历分。

---

### 2. 工作经验

这是一个非常重要的特殊规则。

你需要：

第一步：
识别候选人哪些正式工作经历与JD岗位相关。

第二步：
排除实习经历，不要把实习自动算作正式工作经验。

第三步：
返回relevant_work_records。

第四步：
根据相关正式工作经验与JD要求比较。

工作经验评分原则：

100%以上：
10分

80%-99%：
8分

50%-79%：
6分

25%-49%：
4分

1%-24%：
2分

0：
0分

如果相关性明显偏低，应降低评分。

---

### 3. 岗位相关经验

重点判断“以前实际做过什么”。

不要只看职位名称。

---

### 4. 核心技能

必须寻找简历中的具体使用证据。

例如：

“熟悉Excel”
不能自动证明：
“熟练使用SQL”。

“参与数据分析”
也不能自动证明：
“精通Python数据分析”。

---

### 5. 项目经验

工作经历不能自动转化为项目经验。

只有明确项目、项目行为、项目结果等证据才能获得项目分。

---

### 6. 职责匹配

必须逐条比较JD职责。

每一条职责只能判断：

- 直接
- 部分
- 无

不要用整体印象代替逐项判断。

---

### 最重要

每个维度：

score只能为：

0 / 2 / 4 / 6 / 8 / 10

evidence_grade只能为：

A / B / C / D

如果没有证据：

score必须为0
evidence_grade必须为D

不要为了让候选人看起来合理而补充证据。
"""

    result = call_text_model(
        client,
        model,
        SCORING_SYSTEM_PROMPT,
        prompt
    )

    return safe_json_load(result)


# ============================================================
# 工作经验日期计算
# ============================================================

def parse_year_month(text):
    """
    从：
    2022.07
    2022-07
    2022/07
    2022年07月

    中提取年月。
    """

    if not text:
        return None

    text = str(text)

    match = re.search(
        r"(20\d{2})\s*[.\-/年]\s*(\d{1,2})",
        text
    )

    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2))

    if month < 1 or month > 12:
        return None

    return year, month


def calculate_months(start_text, end_text):
    """
    计算两个年月之间的月数。
    """

    start = parse_year_month(start_text)

    if not start:
        return 0

    if end_text and any(
        x in str(end_text)
        for x in ["至今", "现在", "目前"]
    ):
        now = datetime.now()

        end_year = now.year
        end_month = now.month

    else:
        end = parse_year_month(end_text)

        if not end:
            return 0

        end_year, end_month = end

    start_year, start_month = start

    months = (
        (end_year - start_year) * 12
        + (end_month - start_month)
        + 1
    )

    return max(months, 0)


def calculate_months_from_range(time_text):
    """
    直接从：

    2022.07-2024.06
    2022.07—至今
    2022年7月-2024年6月

    中计算月份。
    """

    if not time_text:
        return 0

    text = str(time_text)

    dates = re.findall(
        r"20\d{2}\s*[.\-/年]\s*\d{1,2}",
        text
    )

    if len(dates) >= 2:

        return calculate_months(
            dates[0],
            dates[1]
        )

    if len(dates) == 1 and any(
        x in text
        for x in ["至今", "现在", "目前"]
    ):

        return calculate_months(
            dates[0],
            "至今"
        )

    return 0


# ============================================================
# 工作经验重新计算
# ============================================================

def recalculate_work_experience(
    score_data,
    jd_data,
    resume_data
):

    required_months = int(
        jd_data.get(
            "required_experience_months",
            0
        ) or 0
    )

    # 如果JD没有明确经验要求，
    # 则不通过月份重新判定。
    if required_months <= 0:

        return {
            "months": 0,
            "required_months": 0,
            "ratio": 0,
            "score": normalize_score(
                score_data.get("score", 0)
            )
        }

    records = score_data.get(
        "relevant_work_records",
        []
    )

    total_months = 0

    valid_records = []

    for item in records:

        if not isinstance(item, dict):
            continue

        # 实习不计入正式工作经验
        if item.get("是否实习") is True:
            continue

        time_text = item.get("时间", "")

        months = calculate_months_from_range(
            time_text
        )

        if months <= 0:
            continue

        relevance = str(
            item.get("相关性", "低")
        )

        # 只有中/高相关才计入岗位相关正式工作经验
        if relevance not in ["高", "中"]:
            continue

        total_months += months

        valid_records.append(
            {
                "公司": item.get("公司", ""),
                "职位": item.get("职位", ""),
                "时间": time_text,
                "months": months,
                "相关性": relevance
            }
        )

    ratio = (
        total_months / required_months
        if required_months > 0
        else 0
    )

    # -----------------------------
    # 根据经验比例确定基础分
    # -----------------------------

    if total_months <= 0:
        score = 0

    elif ratio >= 1:
        score = 10

    elif ratio >= 0.8:
        score = 8

    elif ratio >= 0.5:
        score = 6

    elif ratio >= 0.25:
        score = 4

    else:
        score = 2

    # -----------------------------
    # 根据相关性进行限制
    # -----------------------------

    relevance_values = [
        x["相关性"]
        for x in valid_records
    ]

    if relevance_values:

        if "高" in relevance_values:
            relevance_level = "高"

        else:
            relevance_level = "中"

    else:
        relevance_level = "低"

    if relevance_level == "中":
        score = min(score, 8)

    if relevance_level == "低":
        score = min(score, 4)

    evidence_grade = score_data.get(
        "evidence_grade",
        "D"
    )

    score = apply_evidence_cap(
        score,
        evidence_grade
    )

    return {
        "months": total_months,
        "required_months": required_months,
        "ratio": ratio,
        "score": score,
        "valid_records": valid_records,
        "relevance_level": relevance_level
    }


# ============================================================
# 职责匹配重新校验
# ============================================================

def calculate_duty_coverage(
    jd_data,
    score_data
):

    duties = jd_data.get(
        "岗位职责",
        []
    )

    analysis = score_data.get(
        "duty_analysis",
        []
    )

    if not duties:
        return {
            "coverage": 0,
            "score": 0
        }

    result_map = {}

    for item in analysis:

        if not isinstance(item, dict):
            continue

        duty = str(
            item.get("jd_duty", "")
        ).strip()

        match = str(
            item.get("match", "无")
        ).strip()

        if duty:
            result_map[duty] = match

    matched_score = 0

    direct_count = 0
    partial_count = 0

    for duty in duties:

        match = result_map.get(
            duty,
            "无"
        )

        if match == "直接":
            matched_score += 1
            direct_count += 1

        elif match == "部分":
            matched_score += 0.5
            partial_count += 1

    coverage = matched_score / len(duties)

    # 固定分档
    if coverage >= 0.90:
        score = 10

    elif coverage >= 0.75:
        score = 8

    elif coverage >= 0.50:
        score = 6

    elif coverage >= 0.25:
        score = 4

    elif coverage > 0:
        score = 2

    else:
        score = 0

    # 如果90%以上但是存在明确核心职责完全没有证据，
    # 模型给10也需要谨慎。
    if score == 10:

        for duty in duties:

            match = result_map.get(
                duty,
                "无"
            )

            if match == "无":
                score = 8
                break

    score = apply_evidence_cap(
        score,
        score_data.get(
            "evidence_grade",
            "D"
        )
    )

    return {
        "coverage": coverage,
        "score": score,
        "direct_count": direct_count,
        "partial_count": partial_count,
        "total_count": len(duties)
    }


# ============================================================
# 技能匹配重新校验
# ============================================================

def calculate_skill_coverage(
    jd_data,
    score_data
):

    skills = jd_data.get(
        "技能要求",
        []
    )

    matched = score_data.get(
        "matched_skills",
        []
    )

    unmatched = score_data.get(
        "unmatched_skills",
        []
    )

    total = len(skills)

    if total == 0:

        return {
            "coverage": 0,
            "score": 0
        }

    matched_count = len(matched)

    coverage = matched_count / total

    if coverage >= 0.90:
        score = 10

    elif coverage >= 0.75:
        score = 8

    elif coverage >= 0.50:
        score = 6

    elif coverage >= 0.25:
        score = 4

    elif coverage > 0:
        score = 2

    else:
        score = 0

    score = apply_evidence_cap(
        score,
        score_data.get(
            "evidence_grade",
            "D"
        )
    )

    return {
        "coverage": coverage,
        "score": score,
        "matched_count": matched_count,
        "total_count": total
    }


# ============================================================
# 项目经验重新校验
# ============================================================

def calculate_project_score(
    jd_data,
    score_data
):

    requirements = jd_data.get(
        "项目经验要求",
        []
    )

    matched_projects = score_data.get(
        "matched_projects",
        []
    )

    # JD没有项目要求
    # 这里不强制给予高分
    if not requirements:

        score = normalize_score(
            score_data.get("score", 0)
        )

        score = apply_evidence_cap(
            score,
            score_data.get(
                "evidence_grade",
                "D"
            )
        )

        return {
            "score": score,
            "coverage": None
        }

    if not matched_projects:

        score = 0

    else:

        # 如果JD有明确项目要求，
        # 根据匹配项目证据进行分档
        ratio = min(
            len(matched_projects)
            / len(requirements),
            1
        )

        if ratio >= 0.90:
            score = 10

        elif ratio >= 0.75:
            score = 8

        elif ratio >= 0.50:
            score = 6

        elif ratio >= 0.25:
            score = 4

        else:
            score = 2

    score = apply_evidence_cap(
        score,
        score_data.get(
            "evidence_grade",
            "D"
        )
    )

    return {
        "score": score,
        "coverage": (
            len(matched_projects)
            / len(requirements)
            if requirements
            else None
        )
    }


# ============================================================
# 总评分
# ============================================================

def calculate_final_score(
    scores,
    weights
):

    total = 0

    for dimension in WEIGHTS_DEFAULT.keys():

        score = scores.get(
            dimension,
            0
        )

        weight = weights.get(
            dimension,
            0
        )

        total += (
            score
            * weight
            / 10
        )

    return round(total, 2)


# ============================================================
# UI：JD
# ============================================================

st.header("1️⃣ 输入招聘JD")

jd_input_method = st.radio(
    "JD输入方式",
    [
        "直接粘贴文字",
        "上传文件"
    ],
    horizontal=True
)

jd_text = ""
jd_file = None

if jd_input_method == "直接粘贴文字":

    jd_text = st.text_area(
        "招聘JD",
        height=260,
        placeholder="把完整JD粘贴到这里……"
    )

else:

    jd_file = st.file_uploader(
        "上传JD",
        type=[
            "txt",
            "md",
            "docx",
            "pdf",
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp"
        ],
        key="jd_file"
    )


# ============================================================
# UI：简历
# ============================================================

st.header("2️⃣ 上传候选人简历")

resume_files = st.file_uploader(
    "可以一次上传多份简历",
    type=[
        "txt",
        "md",
        "docx",
        "pdf",
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ],
    accept_multiple_files=True,
    key="resume_files"
)


# ============================================================
# UI：AI配置
# ============================================================

st.header("3️⃣ AI模型")

provider_name = st.selectbox(
    "模型供应商",
    list(PROVIDERS.keys())
)

default_model = PROVIDERS[
    provider_name
]["default_model"]

model_name = st.text_input(
    "模型名称",
    value=default_model
)

api_key = st.text_input(
    "API Key",
    type="password"
)


# ============================================================
# UI：权重
# ============================================================

st.header("4️⃣ 评分权重")

st.caption(
    "默认权重：学历10% / 工作经验15% / "
    "岗位相关经验25% / 核心技能20% / "
    "项目经验15% / 职责匹配15%"
)

weights = {}

cols = st.columns(3)

dimensions = list(
    WEIGHTS_DEFAULT.keys()
)

for i, dimension in enumerate(dimensions):

    default_weight = WEIGHTS_DEFAULT[
        dimension
    ]

    with cols[i % 3]:

        weights[dimension] = st.number_input(
            dimension,
            min_value=0,
            max_value=100,
            value=default_weight,
            step=5
        )

weight_total = sum(weights.values())

if weight_total != 100:

    st.warning(
        f"当前权重合计为 {weight_total}%，必须等于100%。"
    )


# ============================================================
# Rubric展示
# ============================================================

with st.expander("📐 查看当前固定 Rubric"):

    for dimension, rules in RUBRIC.items():

        st.markdown(
            f"### {dimension}"
        )

        for score in ALLOWED_SCORES:

            st.markdown(
                f"- **{score}分**：{rules[score]}"
            )

    st.markdown("### 证据等级")

    st.markdown(
        """
- **A**：直接明确证据，最高10分
- **B**：高度合理推断，最高8分
- **C**：间接证据，最高6分
- **D**：无证据，0分
        """
    )


# ============================================================
# 开始分析
# ============================================================

start_button = st.button(
    "🚀 开始AI筛选",
    type="primary",
    use_container_width=True
)


if start_button:

    # -----------------------------
    # 基础检查
    # -----------------------------

    if not api_key:

        st.error("请先填写 API Key。")
        st.stop()

    if weight_total != 100:

        st.error(
            "评分权重必须合计100%。"
        )

        st.stop()

    if jd_input_method == "直接粘贴文字":

        if not jd_text.strip():

            st.error(
                "请先输入JD。"
            )

            st.stop()

    else:

        if jd_file is None:

            st.error(
                "请先上传JD文件。"
            )

            st.stop()

    if not resume_files:

        st.error(
            "请至少上传一份简历。"
        )

        st.stop()


    # ========================================================
    # 临时目录
    # ========================================================

    temp_dir = Path(
        "temp_resume_screening"
    )

    temp_dir.mkdir(
        exist_ok=True
    )


    # ========================================================
    # 创建AI Client
    # ========================================================

    try:

        client = create_client(
            provider_name,
            api_key
        )

    except Exception as e:

        st.error(
            f"AI Client创建失败：{e}"
        )

        st.stop()


    # ========================================================
    # 解析JD
    # ========================================================

    st.header("🔍 JD解析")

    try:

        if jd_input_method == "直接粘贴文字":

            actual_jd_text = jd_text

        else:

            jd_path = (
                temp_dir
                / jd_file.name
            )

            jd_path.write_bytes(
                jd_file.getbuffer()
            )

            jd_content = extract_file_content(
                jd_path,
                client,
                model_name,
                provider_name
            )

            if jd_content["type"] == "text":

                actual_jd_text = jd_content[
                    "text"
                ]

            else:

                with st.spinner(
                    "正在使用视觉模型读取JD……"
                ):

                    vision_result = call_vision_model(
                        client,
                        model_name,
                        """
你是一名招聘JD文字识别专家。

请准确读取图片中的招聘JD。

只输出图片中实际出现的文字。
不要总结。
不要补充不存在的信息。
""",
                        """
请完整识别这份招聘JD图片中的文字。
""",
                        jd_content["images"]
                    )

                    actual_jd_text = vision_result


        with st.spinner(
            "正在结构化分析JD……"
        ):

            jd_data = parse_jd(
                client,
                model_name,
                actual_jd_text
            )


        with st.expander(
            "查看JD结构化结果",
            expanded=False
        ):

            st.json(jd_data)

    except Exception as e:

        st.error(
            f"JD解析失败：{e}"
        )

        st.stop()


    # ========================================================
    # 简历分析
    # ========================================================

    st.header("📋 候选人分析")

    all_results = []

    progress = st.progress(0)

    total_resumes = len(
        resume_files
    )

    for index, uploaded_file in enumerate(
        resume_files
    ):

        st.subheader(
            f"{index + 1}. {uploaded_file.name}"
        )

        try:

            file_path = (
                temp_dir
                / uploaded_file.name
            )

            file_path.write_bytes(
                uploaded_file.getbuffer()
            )

            # -----------------------------
            # 读取简历
            # -----------------------------

            with st.spinner(
                f"正在读取 {uploaded_file.name}……"
            ):

                resume_content = extract_file_content(
                    file_path,
                    client,
                    model_name,
                    provider_name
                )


            # -----------------------------
            # OCR / Vision
            # -----------------------------

            if resume_content["type"] == "vision":

                with st.spinner(
                    f"正在视觉解析 {uploaded_file.name}……"
                ):

                    vision_text = call_vision_model(
                        client,
                        model_name,
                        """
你是一名简历OCR与信息提取专家。

请完整读取图片中的简历。

必须尽量保留：

- 姓名
- 教育经历
- 专业
- 工作经历
- 公司
- 职位
- 起止时间
- 工作内容
- 项目经历
- 技能

只识别图片中真实存在的信息。
不要自行补充。
不要评价候选人。
""",
                        """
请完整读取这份简历图片中的文字，
按照正常阅读顺序输出。
""",
                        resume_content["images"]
                    )

                    resume_text = vision_text

            else:

                resume_text = resume_content[
                    "text"
                ]


            # -----------------------------
            # 简历结构化
            # -----------------------------

            with st.spinner(
                f"正在结构化 {uploaded_file.name}……"
            ):

                resume_data = parse_resume(
                    client,
                    model_name,
                    resume_text
                )


            # -----------------------------
            # AI初始评分
            # -----------------------------

            with st.spinner(
                f"正在按照Rubric评分 {uploaded_file.name}……"
            ):

                ai_scores = score_candidate(
                    client,
                    model_name,
                    jd_data,
                    resume_data
                )


            # =================================================
            # Python Rubric二次校验
            # =================================================

            final_dimension_scores = {}

            dimension_details = {}


            # -----------------------------
            # 学历
            # -----------------------------

            edu = ai_scores.get(
                "学历匹配",
                {}
            )

            edu_score = apply_evidence_cap(
                edu.get("score", 0),
                edu.get("evidence_grade", "D")
            )

            final_dimension_scores[
                "学历匹配"
            ] = edu_score

            dimension_details[
                "学历匹配"
            ] = edu


            # -----------------------------
            # 工作经验
            # -----------------------------

            work = ai_scores.get(
                "工作经验",
                {}
            )

            work_result = recalculate_work_experience(
                work,
                jd_data,
                resume_data
            )

            final_dimension_scores[
                "工作经验"
            ] = work_result["score"]

            work["python_validation"] = work_result

            dimension_details[
                "工作经验"
            ] = work


            # -----------------------------
            # 岗位相关经验
            # -----------------------------

            role = ai_scores.get(
                "岗位相关经验",
                {}
            )

            role_score = apply_evidence_cap(
                role.get("score", 0),
                role.get(
                    "evidence_grade",
                    "D"
                )
            )

            final_dimension_scores[
                "岗位相关经验"
            ] = role_score

            dimension_details[
                "岗位相关经验"
            ] = role


            # -----------------------------
            # 核心技能
            # -----------------------------

            skill = ai_scores.get(
                "核心技能",
                {}
            )

            skill_result = calculate_skill_coverage(
                jd_data,
                skill
            )

            final_dimension_scores[
                "核心技能"
            ] = skill_result["score"]

            skill["python_validation"] = skill_result

            dimension_details[
                "核心技能"
            ] = skill


            # -----------------------------
            # 项目经验
            # -----------------------------

            project = ai_scores.get(
                "项目经验",
                {}
            )

            project_result = calculate_project_score(
                jd_data,
                project
            )

            final_dimension_scores[
                "项目经验"
            ] = project_result["score"]

            project["python_validation"] = project_result

            dimension_details[
                "项目经验"
            ] = project


            # -----------------------------
            # 职责匹配
            # -----------------------------

            duty = ai_scores.get(
                "职责匹配",
                {}
            )

            duty_result = calculate_duty_coverage(
                jd_data,
                duty
            )

            final_dimension_scores[
                "职责匹配"
            ] = duty_result["score"]

            duty["python_validation"] = duty_result

            dimension_details[
                "职责匹配"
            ] = duty


            # =================================================
            # 最终加权
            # =================================================

            final_score = calculate_final_score(
                final_dimension_scores,
                weights
            )


            # =================================================
            # 保存
            # =================================================

            result = {

                "candidate_file":
                    uploaded_file.name,

                "candidate_name":
                    resume_data.get(
                        "姓名",
                        ""
                    ),

                "jd":
                    jd_data,

                "resume":
                    resume_data,

                "ai_scores":
                    ai_scores,

                "final_dimension_scores":
                    final_dimension_scores,

                "dimension_details":
                    dimension_details,

                "weights":
                    weights,

                "final_score":
                    final_score
            }

            all_results.append(
                result
            )


            # =================================================
            # 当前候选人结果
            # =================================================

            st.markdown(
                f"### 最终得分：**{final_score} / 10**"
            )

            table_lines = [
                "| 维度 | 得分 | 权重 | 加权贡献 | 证据等级 |",
                "|---|---:|---:|---:|---|"
            ]

            for dimension in dimensions:

                score = final_dimension_scores[
                    dimension
                ]

                weight = weights[
                    dimension
                ]

                contribution = round(
                    score * weight / 10,
                    2
                )

                evidence = dimension_details[
                    dimension
                ].get(
                    "evidence_grade",
                    "D"
                )

                table_lines.append(
                    f"| {dimension} | "
                    f"{score} | "
                    f"{weight}% | "
                    f"{contribution} | "
                    f"{evidence} |"
                )

            st.markdown(
                "\n".join(table_lines)
            )


            with st.expander(
                "🔎 查看评分证据"
            ):

                for dimension in dimensions:

                    detail = dimension_details[
                        dimension
                    ]

                    st.markdown(
                        f"#### {dimension}："
                        f"{final_dimension_scores[dimension]}分"
                    )

                    st.write(
                        f"证据等级："
                        f"**{detail.get('evidence_grade', 'D')}**"
                    )

                    st.write(
                        "证据：",
                        detail.get(
                            "evidence",
                            "无"
                        )
                    )

                    st.write(
                        "评分理由：",
                        detail.get(
                            "reason",
                            ""
                        )
                    )

                    if "python_validation" in detail:

                        st.json(
                            detail[
                                "python_validation"
                            ]
                        )


            progress.progress(
                (index + 1)
                / total_resumes
            )


        except Exception as e:

            st.error(
                f"{uploaded_file.name} 分析失败：{e}"
            )

            continue


    # ========================================================
    # 排名
    # ========================================================

    if not all_results:

        st.error(
            "没有成功完成任何候选人的分析。"
        )

        st.stop()


    st.header("🏆 候选人排名")

    sorted_results = sorted(
        all_results,
        key=lambda x: x["final_score"],
        reverse=True
    )


    ranking_lines = [
        "| 排名 | 候选人 | 最终得分 | 学历 | 工作经验 | 岗位相关 | 技能 | 项目 | 职责 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|"
    ]


    for rank, result in enumerate(
        sorted_results,
        start=1
    ):

        scores = result[
            "final_dimension_scores"
        ]

        name = (
            result.get("candidate_name")
            or result["candidate_file"]
        )

        ranking_lines.append(
            f"| {rank} | "
            f"{name} | "
            f"{result['final_score']} | "
            f"{scores['学历匹配']} | "
            f"{scores['工作经验']} | "
            f"{scores['岗位相关经验']} | "
            f"{scores['核心技能']} | "
            f"{scores['项目经验']} | "
            f"{scores['职责匹配']} |"
        )


    st.markdown(
        "\n".join(ranking_lines)
    )


    # ========================================================
    # JSON导出
    # ========================================================

    st.header("📦 导出分析结果")

    export_data = {
        "rubric": RUBRIC,
        "evidence_max_score": EVIDENCE_MAX_SCORE,
        "weights": weights,
        "jd": jd_data,
        "results": sorted_results
    }

    json_bytes = json.dumps(
        export_data,
        ensure_ascii=False,
        indent=2
    ).encode("utf-8")


    st.download_button(
        "⬇️ 下载完整JSON结果",
        data=json_bytes,
        file_name="resume_screening_results.json",
        mime="application/json"
    )


    st.success(
        "分析完成。"
        "当前评分已经经过 Rubric + Python 证据等级封顶。"
    )
