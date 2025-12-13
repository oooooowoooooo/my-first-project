import json
import os
from datetime import datetime, timedelta
from utils.logger import logger
from utils.time_utils import get_date_str, get_past_n_days


class RewardSystem:
    def __init__(self, reward_config="config/rewards_config.json", task_config="config/tasks_config.json"):
        # 加载配置（容错处理）
        self.reward_config = {"punishments": {}, "rewards": {}, "level_system": {}}
        self.task_config = {"daily_tasks": []}

        try:
            with open(reward_config, "r", encoding="utf-8") as f:
                self.reward_config = json.load(f)
        except Exception as e:
            logger.error(f"加载奖惩配置失败：{e}")

        try:
            with open(task_config, "r", encoding="utf-8") as f:
                self.task_config = json.load(f)
        except Exception as e:
            logger.error(f"加载任务配置失败：{e}")

        self.records_path = "data/task_records.json"
        self._init_records_file()

    def _init_records_file(self):
        """初始化记录文件"""
        try:
            if not os.path.exists("data"):
                os.makedirs("data")
            if not os.path.exists(self.records_path):
                with open(self.records_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"初始化记录文件失败：{e}")

    def update_points(self, date, points, reason):
        """
        更新积分（原子操作）
        :param date: 日期
        :param points: 积分值（可正可负）
        :param reason: 原因
        :return: bool - 是否成功
        """
        try:
            # 读取现有记录
            records = {}
            try:
                with open(self.records_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except:
                records = {}

            # 初始化当日记录
            if date not in records:
                records[date] = {"calories": {}, "tasks": {}, "points": 0}

            # 更新积分
            old_points = records[date]["points"]
            records[date]["points"] += points
            new_points = records[date]["points"]

            logger.info(f"积分更新：{old_points} + {points} = {new_points} | 原因：{reason}")

            # 原子保存（避免文件损坏）
            temp_path = f"{self.records_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.records_path)

            return True
        except Exception as e:
            logger.error(f"更新积分失败：{e}", exc_info=True)
            return False

    def check_daily_rewards(self, date=None):
        """检查并执行当日奖惩"""
        if date is None:
            date = get_date_str()

        # 读取记录
        records = {}
        try:
            with open(self.records_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as e:
            logger.error(f"读取记录失败：{e}")
            return

        if date not in records:
            logger.info(f"无{date}的记录，跳过奖惩结算")
            return

        # 1. 检查未完成必做任务惩罚
        task_records = records[date]["tasks"]
        required_tasks = [t for t in self.task_config["daily_tasks"] if t.get("required", False)]
        missed_tasks = []

        for task in required_tasks:
            task_id = task["task_id"]
            if task_id not in task_records or not task_records[task_id].get("completed", False):
                missed_tasks.append(task["task_name"])
                # 执行惩罚
                self.update_points(
                    date,
                    self.reward_config["punishments"]["miss_required_task"]["points"],
                    self.reward_config["punishments"]["miss_required_task"]["message"]
                )

        # 2. 多个未完成任务额外惩罚
        threshold = self.reward_config["punishments"]["multiple_misses"]["threshold"]
        if len(missed_tasks) >= threshold:
            self.update_points(
                date,
                self.reward_config["punishments"]["multiple_misses"]["points"],
                self.reward_config["punishments"]["multiple_misses"]["message"]
            )

        # 3. 卡路里超标惩罚
        try:
            from core.calorie_tracker import CalorieTracker
            calorie_tracker = CalorieTracker()
            calorie_summary = calorie_tracker.get_daily_calorie_summary(date)
            if not calorie_summary["within_limit"] and calorie_summary["total"] > 0:
                self.update_points(
                    date,
                    self.reward_config["punishments"]["exceed_calorie_limit"]["points"],
                    self.reward_config["punishments"]["exceed_calorie_limit"]["message"]
                )
        except Exception as e:
            logger.error(f"检查卡路里奖惩失败：{e}")

        # 4. 完成所有任务奖励
        if len(missed_tasks) == 0 and len(required_tasks) > 0:
            self.update_points(
                date,
                self.reward_config["rewards"]["complete_all_tasks"]["points"],
                self.reward_config["rewards"]["complete_all_tasks"]["message"]
            )

        # 5. 卡路里达标奖励
        try:
            if calorie_summary["within_limit"] and calorie_summary["total"] > 0:
                self.update_points(
                    date,
                    self.reward_config["rewards"]["calorie_within_limit"]["points"],
                    self.reward_config["rewards"]["calorie_within_limit"]["message"]
                )
        except:
            pass

        # 6. 连续完成奖励
        self.check_streak_rewards(date)

    def check_streak_rewards(self, date=None):
        """检查连续完成奖励"""
        if date is None:
            date = get_date_str()

        # 获取过去3天日期
        past_3_days = get_past_n_days(3)

        # 检查每天是否完成所有任务
        streak_count = 0
        try:
            with open(self.records_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            required_tasks = [t for t in self.task_config["daily_tasks"] if t.get("required", False)]
            if len(required_tasks) == 0:
                return

            for day in past_3_days:
                if day not in records:
                    continue

                # 检查当日是否完成所有必做任务
                task_records = records[day]["tasks"]
                all_completed = True
                for task in required_tasks:
                    if task["task_id"] not in task_records or not task_records[task["task_id"]]["completed"]:
                        all_completed = False
                        break

                if all_completed:
                    streak_count += 1
                else:
                    break  # 连续中断

            # 连续3天完成奖励
            if streak_count >= 3:
                self.update_points(
                    date,
                    self.reward_config["rewards"]["streak_3_days"]["points"],
                    self.reward_config["rewards"]["streak_3_days"]["message"]
                )
                logger.info(f"连续{streak_count}天完成任务，发放奖励")
        except Exception as e:
            logger.error(f"检查连续奖励失败：{e}")

    def get_user_level(self, date=None):
        """获取用户当前等级（容错处理）"""
        # 计算总积分
        total_points = 0
        try:
            with open(self.records_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            for day in records:
                total_points += records[day].get("points", 0)
        except Exception as e:
            logger.error(f"计算总积分失败：{e}")
            total_points = 0

        # 匹配等级（容错处理）
        level_system = self.reward_config.get("level_system", {})
        if not level_system:
            # 使用默认等级
            level_system = {
                "1": {"min_points": 0, "max_points": 50, "name": "新手"},
                "2": {"min_points": 51, "max_points": 150, "name": "进阶者"},
                "3": {"min_points": 151, "max_points": 300, "name": "自律达人"},
                "4": {"min_points": 301, "max_points": 500, "name": "超级自律者"},
                "5": {"min_points": 501, "max_points": 1000, "name": "自律大师"}
            }

        # 查找当前等级
        current_level = "1"
        current_level_info = level_system["1"]

        for level_id, level_info in level_system.items():
            if level_info["min_points"] <= total_points <= level_info["max_points"]:
                current_level = level_id
                current_level_info = level_info

        # 计算下一等级
        level_ids = sorted([int(k) for k in level_system.keys()])
        current_level_int = int(current_level)
        next_level_idx = level_ids.index(current_level_int) + 1

        if next_level_idx < len(level_ids):
            next_level = str(level_ids[next_level_idx])
        else:
            next_level = "最高等级"

        return {
            "level": current_level,
            "name": current_level_info["name"],
            "total_points": total_points,
            "next_level": next_level
        }