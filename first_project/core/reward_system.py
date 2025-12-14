import json
import os
from datetime import datetime, timedelta
from utils.logger import logger
from utils.time_utils import get_date_str, get_past_n_days


class RewardSystem:
    def __init__(self, reward_config="config/rewards_config.json", task_config="config/tasks_config.json"):
        # 加载配置
        with open(reward_config, "r", encoding="utf-8") as f:
            self.reward_config = json.load(f)
        with open(task_config, "r", encoding="utf-8") as f:
            self.task_config = json.load(f)

        self.records_path = "data/task_records.json"
        self._init_records_file()

    def _init_records_file(self):
        """初始化记录文件，确保结构完整"""
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists(self.records_path):
            # 初始化完整的记录结构
            init_data = {
                "user_info": {"total_points": 0, "current_level": "1", "streak_days": 0},
                "daily_records": {}
            }
            with open(self.records_path, "w", encoding="utf-8") as f:
                json.dump(init_data, f, ensure_ascii=False, indent=2)

    def _get_records(self):
        """读取记录（统一方法，确保数据结构正确）"""
        with open(self.records_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        # 兼容旧数据结构
        if "user_info" not in records:
            records["user_info"] = {"total_points": 0, "current_level": "1", "streak_days": 0}
        if "daily_records" not in records:
            records["daily_records"] = {}

        return records

    def _save_records(self, records):
        """保存记录（原子操作）"""
        temp_path = f"{self.records_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.records_path)

    def update_points(self, points, reason, date=None):
        """
        真正生效的积分更新：更新总积分+当日积分
        :param points: 积分值（可正可负）
        :param reason: 变动原因
        :param date: 日期（默认今日）
        """
        if date is None:
            date = get_date_str()

        try:
            records = self._get_records()

            # 初始化当日记录
            if date not in records["daily_records"]:
                records["daily_records"][date] = {
                    "calories": {},
                    "tasks": {},
                    "points_change": [],
                    "daily_points": 0
                }

            # 记录当日积分变动
            records["daily_records"][date]["points_change"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "points": points,
                "reason": reason
            })
            # 更新当日总积分
            records["daily_records"][date]["daily_points"] += points
            # 更新用户总积分
            records["user_info"]["total_points"] += points

            # 保存记录
            self._save_records(records)

            logger.info(f"积分更新：总积分 {records['user_info']['total_points']} | 当日变动 {points} | 原因：{reason}")
            return True
        except Exception as e:
            logger.error(f"更新积分失败：{e}")
            return False

    def check_daily_rewards(self, date=None):
        """真正生效的每日奖惩结算"""
        if date is None:
            date = get_date_str()

        records = self._get_records()
        if date not in records["daily_records"]:
            logger.warning(f"{date} 无任务记录，跳过结算")
            return

        # 1. 获取当日任务完成情况
        task_records = records["daily_records"][date]["tasks"]
        required_tasks = [t for t in self.task_config["daily_tasks"] if t["required"]]
        missed_tasks = []

        # 检查未完成的必做任务
        for task in required_tasks:
            task_id = str(task["task_id"])
            if task_id not in task_records or not task_records[task_id]["completed"]:
                missed_tasks.append(task["task_name"])
                # 未完成惩罚（真正扣积分）
                self.update_points(
                    self.reward_config["punishments"]["miss_required_task"]["points"],
                    self.reward_config["punishments"]["miss_required_task"]["message"]
                )

        # 2. 多个未完成任务额外惩罚
        if len(missed_tasks) >= self.reward_config["punishments"]["multiple_misses"]["threshold"]:
            self.update_points(
                self.reward_config["punishments"]["multiple_misses"]["points"],
                self.reward_config["punishments"]["multiple_misses"]["message"]
            )

        # 3. 卡路里超标惩罚
        from core.calorie_tracker import CalorieTracker
        calorie_tracker = CalorieTracker()
        calorie_summary = calorie_tracker.get_daily_calorie_summary(date)
        if not calorie_summary["within_limit"] and calorie_summary["total"] > 0:
            self.update_points(
                self.reward_config["punishments"]["exceed_calorie_limit"]["points"],
                self.reward_config["punishments"]["exceed_calorie_limit"]["message"]
            )

        # 4. 完成所有任务奖励
        if len(missed_tasks) == 0 and len(required_tasks) > 0:
            self.update_points(
                self.reward_config["rewards"]["complete_all_tasks"]["points"],
                self.reward_config["rewards"]["complete_all_tasks"]["message"]
            )

        # 5. 卡路里达标奖励
        if calorie_summary["within_limit"] and calorie_summary["total"] > 0:
            self.update_points(
                self.reward_config["rewards"]["calorie_within_limit"]["points"],
                self.reward_config["rewards"]["calorie_within_limit"]["message"]
            )

        # 6. 连续完成奖励
        self.check_streak_rewards(date)

        # 7. 更新用户等级
        self.update_user_level()

    def check_streak_rewards(self, date=None):
        """修复连续天数统计逻辑"""
        if date is None:
            date = get_date_str()

        records = self._get_records()
        # 获取过去3天日期
        past_3_days = get_past_n_days(3)
        streak_count = 0

        # 统计连续完成天数
        for day in past_3_days:
            if day not in records["daily_records"]:
                break

            day_tasks = records["daily_records"][day]["tasks"]
            required_tasks = [t for t in self.task_config["daily_tasks"] if t["required"]]
            all_completed = True

            for task in required_tasks:
                task_id = str(task["task_id"])
                if task_id not in day_tasks or not day_tasks[task_id]["completed"]:
                    all_completed = False
                    break

            if all_completed:
                streak_count += 1
            else:
                break

        # 更新连续天数
        records["user_info"]["streak_days"] = streak_count
        self._save_records(records)

        # 连续3天奖励
        if streak_count >= 3:
            self.update_points(
                self.reward_config["rewards"]["streak_3_days"]["points"],
                self.reward_config["rewards"]["streak_3_days"]["message"]
            )

    def update_user_level(self):
        """根据总积分更新用户等级（真正生效）"""
        records = self._get_records()
        total_points = records["user_info"]["total_points"]
        level_system = self.reward_config["level_system"]

        # 匹配等级
        current_level = "1"
        for level_id, level_info in level_system.items():
            if level_info["min_points"] <= total_points <= level_info["max_points"]:
                current_level = level_id

        # 更新等级
        records["user_info"]["current_level"] = current_level
        self._save_records(records)
        return current_level

    def get_user_level(self):
        """获取用户当前等级（含总积分）"""
        records = self._get_records()
        total_points = records["user_info"]["total_points"]
        current_level = records["user_info"]["current_level"]
        level_system = self.reward_config["level_system"]

        # 获取等级详情
        level_info = level_system.get(current_level, level_system["1"])
        level_ids = sorted([int(k) for k in level_system.keys()])
        current_level_int = int(current_level)

        # 计算下一等级
        next_level = "最高等级"
        if current_level_int < max(level_ids):
            next_level = str(current_level_int + 1)

        return {
            "level": current_level,
            "name": level_info["name"],
            "total_points": total_points,
            "streak_days": records["user_info"]["streak_days"],
            "next_level": next_level
        }

    def get_daily_points(self, date=None):
        """获取当日积分变动详情"""
        if date is None:
            date = get_date_str()

        records = self._get_records()
        if date not in records["daily_records"]:
            return {"daily_points": 0, "points_change": []}

        return {
            "daily_points": records["daily_records"][date]["daily_points"],
            "points_change": records["daily_records"][date]["points_change"]
        }