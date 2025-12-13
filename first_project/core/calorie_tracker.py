import json
import os
from datetime import datetime
from utils.logger import logger
from utils.time_utils import get_date_str


class CalorieTracker:
    def __init__(self, config_path="config/tasks_config.json"):
        # 加载配置（容错处理）
        self.config = {"calorie_limits": {}}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"加载卡路里配置失败：{e}")
            # 使用默认配置
            self.config["calorie_limits"] = {
                "breakfast": 500,
                "lunch": 700,
                "dinner": 600,
                "daily_total": 1800
            }

        self.calorie_limits = self.config["calorie_limits"]
        self.records_path = "data/task_records.json"
        self._init_records_file()

    def _init_records_file(self):
        """初始化任务记录文件（容错处理）"""
        try:
            if not os.path.exists("data"):
                os.makedirs("data")
                logger.info("创建data目录")

            if not os.path.exists(self.records_path):
                with open(self.records_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
                logger.info("创建任务记录文件")
        except Exception as e:
            logger.error(f"初始化记录文件失败：{e}")

    def record_calorie(self, meal_type, calories):
        """
        记录某一餐的卡路里
        :param meal_type: 餐次（breakfast/lunch/dinner）
        :param calories: 卡路里数值（正数）
        :return: bool - 是否成功
        """
        try:
            # 输入验证
            calories = float(calories)
            if calories < 0:
                logger.error(f"卡路里不能为负数：{calories}")
                return False

            date = get_date_str()

            # 读取现有记录（容错处理）
            records = {}
            try:
                with open(self.records_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except:
                records = {}

            # 初始化当日记录
            if date not in records:
                records[date] = {"calories": {}, "tasks": {}, "points": 0}

            # 记录卡路里
            records[date]["calories"][meal_type] = calories
            logger.info(f"记录{meal_type}卡路里：{calories}")

            # 保存记录（原子操作，避免文件损坏）
            temp_path = f"{self.records_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.records_path)  # 原子替换

            # 检查是否超标
            self.check_calorie_limit(date)
            return True
        except ValueError:
            logger.error(f"卡路里必须是数字：{calories}")
            return False
        except Exception as e:
            logger.error(f"记录卡路里失败：{e}", exc_info=True)
            return False

    def check_calorie_limit(self, date=None):
        """检查卡路里是否超标"""
        if date is None:
            date = get_date_str()

        try:
            with open(self.records_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            if date not in records or not records[date]["calories"]:
                return True

            calorie_records = records[date]["calories"]
            total_calories = sum(calorie_records.values())

            # 检查单餐限制
            over_limit = False
            for meal_type in ["breakfast", "lunch", "dinner"]:
                limit = self.calorie_limits.get(meal_type, 0)
                if meal_type in calorie_records and calorie_records[meal_type] > limit:
                    logger.warning(f"{meal_type}卡路里超标：{calorie_records[meal_type]} > {limit}")
                    over_limit = True

            # 检查总热量限制
            total_limit = self.calorie_limits.get("daily_total", 0)
            if total_calories > total_limit:
                logger.warning(f"每日总卡路里超标：{total_calories} > {total_limit}")
                over_limit = True

            return not over_limit
        except Exception as e:
            logger.error(f"检查卡路里限制失败：{e}")
            return True

    def get_daily_calorie_summary(self, date=None):
        """获取当日卡路里汇总"""
        if date is None:
            date = get_date_str()

        # 默认返回值
        summary = {
            "total": 0,
            "meals": {},
            "limits": self.calorie_limits,
            "within_limit": True
        }

        try:
            with open(self.records_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            if date in records and "calories" in records[date]:
                calorie_records = records[date]["calories"]
                summary["meals"] = calorie_records
                summary["total"] = sum(calorie_records.values())
                # 检查是否超标
                summary["within_limit"] = self.check_calorie_limit(date)
        except Exception as e:
            logger.error(f"获取卡路里汇总失败：{e}")

        return summary