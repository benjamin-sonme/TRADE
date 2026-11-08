"""
Universe Module
Contains curated lists of major market benchmarks and candidate universes,
including the Top 50 Nasdaq companies by market capitalization.
"""

from typing import Dict, List

# Nasdaq Top 50 by Market Capitalization
NASDAQ_TOP_50_MAP: Dict[str, Dict[str, str]] = {
    "AAPL": {"name": "Apple (苹果)", "sector": "消费电子 / 科技"},
    "MSFT": {"name": "Microsoft (微软)", "sector": "软件 / 云计算"},
    "NVDA": {"name": "Nvidia (英伟达)", "sector": "半导体 / AI算力"},
    "AMZN": {"name": "Amazon (亚马逊)", "sector": "电商 / 云计算"},
    "GOOGL": {"name": "Alphabet Class A (谷歌-A)", "sector": "互联网 / AI"},
    "GOOG": {"name": "Alphabet Class C (谷歌-C)", "sector": "互联网 / AI"},
    "META": {"name": "Meta Platforms (脸书/Meta)", "sector": "社交媒体 / 元宇宙"},
    "TSLA": {"name": "Tesla (特斯拉)", "sector": "新能源汽车 / 自动驾驶"},
    "AVGO": {"name": "Broadcom (博通)", "sector": "半导体 / 通信芯片"},
    "COST": {"name": "Costco (开市客)", "sector": "大型零售 / 必需消费"},
    "ASML": {"name": "ASML Holding (阿斯麦)", "sector": "半导体光刻机设备"},
    "PEP": {"name": "PepsiCo (百事可乐)", "sector": "饮料与食品 / 必需消费"},
    "AZN": {"name": "AstraZeneca (阿斯利康)", "sector": "生物制药 / 医疗"},
    "CSCO": {"name": "Cisco (思科)", "sector": "网络与企业硬件"},
    "TMUS": {"name": "T-Mobile US (T移动)", "sector": "电信服务"},
    "LIN": {"name": "Linde (林德集团)", "sector": "基础化工 / 工业气体"},
    "AMD": {"name": "Advanced Micro Devices (超威半导体)", "sector": "芯片 / GPU/CPU"},
    "NFLX": {"name": "Netflix (奈飞)", "sector": "流媒体 / 娱乐内容"},
    "QCOM": {"name": "Qualcomm (高通)", "sector": "移动无线芯片 / 通信"},
    "TXN": {"name": "Texas Instruments (德州仪器)", "sector": "模拟半导体 / 工业芯片"},
    "AMGN": {"name": "Amgen (安进)", "sector": "生物制药"},
    "INTU": {"name": "Intuit (财捷)", "sector": "企业财税软件"},
    "ISRG": {"name": "Intuitive Surgical (直觉外科)", "sector": "达芬奇手术机器人"},
    "HON": {"name": "Honeywell (霍尼韦尔)", "sector": "航空航天 / 工业制造"},
    "BKNG": {"name": "Booking Holdings (缤客)", "sector": "在线旅游与酒店预订"},
    "VRTX": {"name": "Vertex Pharma (福泰制药)", "sector": "基因与生物制药"},
    "SBUX": {"name": "Starbucks (星巴克)", "sector": "连锁餐饮 / 消费"},
    "CMCSA": {"name": "Comcast (康卡斯特)", "sector": "电信 / 传媒娱乐"},
    "GILD": {"name": "Gilead Sciences (吉利德科学)", "sector": "抗病毒生物制药"},
    "MDLZ": {"name": "Mondelez (亿滋国际)", "sector": "休闲食品 / 必需消费"},
    "REGN": {"name": "Regeneron (再生元制药)", "sector": "生物制药"},
    "PANW": {"name": "Palo Alto Networks (派拓网络)", "sector": "企业级网络安全"},
    "AMAT": {"name": "Applied Materials (应用材料)", "sector": "半导体制造设备"},
    "ADI": {"name": "Analog Devices (亚德诺半导体)", "sector": "高性能模拟芯片"},
    "LRCX": {"name": "Lam Research (泛林半导体)", "sector": "半导体蚀刻设备"},
    "MU": {"name": "Micron Technology (美光科技)", "sector": "存储芯片 (DRAM/NAND)"},
    "KLAC": {"name": "KLA Corp (科磊)", "sector": "半导体检测与量测设备"},
    "SNPS": {"name": "Synopsys (新思科技)", "sector": "EDA电子设计自动化软件"},
    "CDNS": {"name": "Cadence (铿腾电子)", "sector": "EDA电子设计自动化软件"},
    "MELI": {"name": "MercadoLibre (美客多)", "sector": "拉美电商与数字金融"},
    "CRWD": {"name": "CrowdStrike (众击)", "sector": "端点安全 / 云安全"},
    "ABNB": {"name": "Airbnb (爱彼迎)", "sector": "共享住宿 / 在线旅游"},
    "MAR": {"name": "Marriott (万豪国际)", "sector": "酒店连锁与文旅"},
    "CTAS": {"name": "Cintas (信达思)", "sector": "工业与企业服务"},
    "PYPL": {"name": "PayPal (贝宝)", "sector": "数字支付与金融科技"},
    "ORLY": {"name": "O'Reilly Automotive (奥莱利)", "sector": "汽车后市场零售"},
    "NXPI": {"name": "NXP Semiconductors (恩智浦)", "sector": "汽车电子与微控制器"},
    "FTNT": {"name": "Fortinet (飞塔)", "sector": "防火墙 / 网络安全"},
    "CEG": {"name": "Constellation Energy (星座能源)", "sector": "清洁核能 / AI电力基础设施"},
    "SPCX": {"name": "SpaceX (太空探索技术)", "sector": "商业航天 / 星链卫星"},
}

NASDAQ_TOP_50_TICKERS: List[str] = list(NASDAQ_TOP_50_MAP.keys())


def print_nasdaq_top50_list():
    """Prints a neat formatted table of Nasdaq Top 50 companies."""
    print("\n" + "=" * 75)
    print(" 🏛️ 纳斯达克 (NASDAQ) 市值前 50 强标的备选池")
    print("=" * 75)
    for idx, (ticker, info) in enumerate(NASDAQ_TOP_50_MAP.items(), 1):
        print(f" {idx:2d}. {ticker:<6} | {info['name']:<28} | {info['sector']}")
    print("=" * 75 + "\n")
