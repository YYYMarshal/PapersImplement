from datetime import datetime


def get_current_time():
    """
    显示当前时间的时分秒格式
    """
    current_time = datetime.now()
    formatted_time = current_time.strftime("%H:%M:%S")
    print(f"当前时间：{formatted_time}")
    return current_time


def time_difference(start_time):
    """
    计算当前时间减去给定时间的时间差
    """
    current_time = get_current_time()
    time_diff = current_time - start_time
    print(f"用时：{time_diff}")
    return time_diff
