from datetime import datetime, timedelta

def get_current_time_str():
    """获取当前时间字符串（HH:MM）"""
    return datetime.now().strftime("%H:%M")

def get_date_str(date=None):
    """获取日期字符串（YYYY-MM-DD）"""
    if date is None:
        date = datetime.now()
    return date.strftime("%Y-%m-%d")

def parse_time_str(time_str):
    """解析时间字符串为datetime.time对象（容错处理）"""
    try:
        return datetime.strptime(time_str, "%H:%M").time()
    except ValueError:
        # 兼容其他格式
        try:
            return datetime.strptime(time_str, "%H:%M:%S").time()
        except:
            return datetime.now().time()

def is_time_between(start_time, end_time, check_time=None):
    """检查时间是否在指定区间内"""
    if check_time is None:
        check_time = datetime.now().time()
    try:
        start = parse_time_str(start_time)
        end = parse_time_str(end_time)
        return start <= check_time <= end
    except:
        return False

def get_past_n_days(n):
    """获取过去n天的日期列表（包含今天）"""
    dates = []
    for i in range(n):
        date = datetime.now() - timedelta(days=i)
        dates.append(get_date_str(date))
    return dates