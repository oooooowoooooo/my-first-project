import json
import os
from datetime import datetime
from utils.logger import logger
from utils.time_utils import get_date_str


class TaskManager:
    def __init__(self, config_path="config/tasks_config.json"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.daily_tasks = self.config["daily_tasks"]
        self.records_path = "data/task_records.json"
        self._init_records_file()

    def _init_records_file(self):
        """初始化记录文件（确保完整结构）"""
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
        """统一读取记录"""
        with open(self.records_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        # 兼容旧结构
        if "daily_records" not in records:
            records["daily_records"] = {}
        return records

    def _save_records(self, records):
        """原子保存记录"""
        temp_path = f"{self.records_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.records_path)

    def mark_task_complete(self, task_id, value=None, date=None):
        """
        真正持久化的任务完成标记
        :param task_id: 任务ID
        :param value: 完成值（如学习时长）
        :param date: 日期（默认今日）
        """
        if date is None:
            date = get_date_str()

        try:
            task_id = str(int(task_id))  # 统一为字符串ID
            records = self._get_records()

            # 验证任务ID
            task = next((t for t in self.daily_tasks if str(t["task_id"]) == task_id), None)
            if not task:
                logger.error(f"任务ID {task_id} 不存在")
                return False

            # 初始化当日记录
            if date not in records["daily_records"]:
                records["daily_records"][date] = {
                    "calories": {},
                    "tasks": {},
                    "points_change": [],
                    "daily_points": 0
                }

            # 处理完成值
            saved_value = value
            if saved_value is not None and saved_value != "":
                try:
                    saved_value = float(saved_value)
                except:
                    pass

            # 真正记录任务完成状态
            records["daily_records"][date]["tasks"][task_id] = {
                "task_id": task_id,
                "task_name": task["task_name"],
                "task_type": task["task_type"],
                "completed": True,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "value": saved_value,
                "target_value": task["target_value"]
            }

            # 保存记录
            self._save_records(records)
            logger.info(f"任务标记完成：{task['task_name']} | 值：{saved_value} | 日期：{date}")
            return True
        except Exception as e:
            logger.error(f"标记任务失败：{e}")
            return False

    def get_today_tasks(self, date=None):
        """获取当日任务完成情况（真正读取记录）"""
        if date is None:
            date = get_date_str()

        records = self._get_records()
        task_status = []

        # 遍历所有配置任务，匹配完成状态
        for task in self.daily_tasks:
            task_id = str(task["task_id"])
            task_info = {
                "task_id": task_id,
                "task_name": task["task_name"],
                "task_type": task["task_type"],
                "completed": False,
                "completed_at": None,
                "value": None,
                "target_value": task["target_value"],
                "description": task["description"],
                "required": task["required"],
                "reminder_times": task["reminder_times"]
            }

            # 读取真正的完成状态
            if (date in records["daily_records"] and
                    task_id in records["daily_records"][date]["tasks"]):
                task_record = records["daily_records"][date]["tasks"][task_id]
                task_info["completed"] = task_record["completed"]
                task_info["completed_at"] = task_record["completed_at"]
                task_info["value"] = task_record["value"]

            task_status.append(task_info)

        return task_status

    def get_task_summary(self, date=None):
        """获取任务完成统计（真实数据）"""
        if date is None:
            date = get_date_str()

        tasks = self.get_today_tasks(date)
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t["completed"]])

        # 必做任务统计
        required_tasks = [t for t in tasks if t["required"]]
        completed_required = len([t for t in required_tasks if t["completed"]])

        return {
            "date": date,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_rate": round(completed_tasks / total_tasks * 100, 2) if total_tasks > 0 else 0,
            "required_tasks": len(required_tasks),
            "completed_required": completed_required,
            "required_completion_rate": round(completed_required / len(required_tasks) * 100, 2) if len(
                required_tasks) > 0 else 0
        }

    def get_task_history(self, days=7):
        """获取历史任务完成情况（新增）"""
        history = []
        past_days = get_past_n_days(days)

        for day in past_days:
            summary = self.get_task_summary(day)
            history.append(summary)

        return history