import json
import os
from datetime import datetime
from utils.logger import logger
from utils.time_utils import get_date_str


class CalorieTracker:
    def __init__(self, config_path="config/tasks_config.json"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.calorie_limits = self.config["calorie_limits"]
        self.records_path = "data/task_records.json"
        self._init_records_file()

    def _init_records_file(self):
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists(self.records_path):
            init_data = {
                "user_info": {"total_points": 0, "current_level": "1", "streak_days": 0},
                "daily_records": {}
            }
            with open(self.records_path, "w", encoding="utf-8") as f:
                json.dump(init_data, f, ensure_ascii=False, indent=2)

    def _get_records(self):
        with open(self.records_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if "daily_records" not in records:
            records["daily_records"] = {}
        return records

    def _save_records(self, records):
        temp_path = f"{self.records_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.records_path)

    def record_calorie(self, meal_type, calories, date=None):
        """真正持久化的卡路里记录"""
        if date is None:
            date = get_date_str()

        try:
            calories = float(calories)
            if calories < 0:
                logger.error("卡路里不能为负数")
                return False

            records = self._get_records()

            # 初始化当日记录
            if date not in records["daily_records"]:
                records["daily_records"][date] = {
                    "calories": {},
                    "tasks": {},
                    "points_change": [],
                    "daily_points": 0
                }

            # 真正记录卡路里
            records["daily_records"][date]["calories"][meal_type] = {
                "value": calories,
                "limit": self.calorie_limits[meal_type],
                "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # 保存记录
            self._save_records(records)
            logger.info(f"记录卡路里：{meal_type} {calories} | 日期：{date}")

            # 检查是否超标
            self.check_calorie_limit(date)
            return True
        except ValueError:
            logger.error("卡路里必须是数字")
            return False
        except Exception as e:
            logger.error(f"记录卡路里失败：{e}")
            return False

    def get_daily_calorie_summary(self, date=None):
        """获取当日卡路里汇总（真实数据）"""
        if date is None:
            date = get_date_str()

        records = self._get_records()
        summary = {
            "total": 0,
            "meals": {},
            "limits": self.calorie_limits,
            "within_limit": True
        }

        if date in records["daily_records"] and "calories" in records["daily_records"][date]:
            calorie_records = records["daily_records"][date]["calories"]

            # 计算总卡路里
            total = 0
            meals = {}
            for meal_type, data in calorie_records.items():
                calories = data["value"]
                total += calories
                meals[meal_type] = {
                    "value": calories,
                    "limit": self.calorie_limits[meal_type],
                    "over_limit": calories > self.calorie_limits[meal_type]
                }

            summary["total"] = total
            summary["meals"] = meals

            # 检查是否超标
            if total > self.calorie_limits["daily_total"]:
                summary["within_limit"] = False
            else:
                # 检查单餐超标
                for meal_type, data in meals.items():
                    if data["over_limit"]:
                        summary["within_limit"] = False
                        break

        return summary