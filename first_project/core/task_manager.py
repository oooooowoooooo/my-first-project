import json
import os
from datetime import datetime
from utils.logger import logger
from utils.time_utils import get_date_str


class TaskManager:
    def __init__(self, config_path="config/tasks_config.json"):
        # 加载任务配置（容错处理）
        self.config = {"daily_tasks": []}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"加载任务配置失败：{e}")

        self.daily_tasks = self.config["daily_tasks"]
        self.records_path = "data/task_records.json"
        self._init_records_file()

    def _init_records_file(self):
        """初始化任务记录文件"""
        try:
            if not os.path.exists("data"):
                os.makedirs("data")
            if not os.path.exists(self.records_path):
                with open(self.records_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"初始化记录文件失败：{e}")

    def mark_task_complete(self, task_id, value=None):
        """
        标记任务完成（容错处理）
        :param task_id: 任务ID
        :param value: 任务完成值（如学习时长、卡路里等）
        :return: bool - 是否成功
        """
        try:
            task_id = int(task_id)
            date = get_date_str()

            # 验证任务ID是否存在
            task = next((t for t in self.daily_tasks if t["task_id"] == task_id), None)
            if not task:
                logger.error(f"任务ID {task_id} 不存在")
                return False

            # 读取现有记录
            records = {}
            try:
                with open(self.records_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception as e:
                logger.warning(f"读取记录失败，创建新记录：{e}")
                records = {}

            # 初始化当日记录
            if date not in records:
                records[date] = {"calories": {}, "tasks": {}, "points": 0}

            # 处理完成值（允许0值）
            saved_value = value
            if saved_value is not None and saved_value != "":
                # 尝试转换为数字（数值型任务）
                try:
                    saved_value = float(saved_value)
                except:
                    pass  # 非数值型保持原样

            # 记录任务完成情况
            records[date]["tasks"][task_id] = {
                "task_name": task["task_name"],
                "completed": True,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "value": saved_value
            }

            logger.info(f"标记任务完成：{task['task_name']} | 值：{saved_value}")

            # 原子保存（避免文件损坏）
            temp_path = f"{self.records_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.records_path)

            return True
        except ValueError:
            logger.error(f"任务ID必须是数字：{task_id}")
            return False
        except Exception as e:
            logger.error(f"标记任务完成失败：{e}", exc_info=True)
            return False

    def get_today_tasks(self, date=None):
        """获取当日任务完成情况（容错处理）"""
        if date is None:
            date = get_date_str()

        task_status = []
        try:
            # 读取记录
            records = {}
            try:
                with open(self.records_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except:
                records = {}

            # 构建任务状态
            for task in self.daily_tasks:
                task_id = task["task_id"]
                task_info = {
                    "task_id": task_id,
                    "task_name": task["task_name"],
                    "task_type": task.get("task_type", "routine"),
                    "completed": False,
                    "value": None,
                    "description": task.get("description", ""),
                    "target_value": task.get("target_value")
                }

                # 检查是否已完成
                if date in records and task_id in records[date]["tasks"]:
                    task_record = records[date]["tasks"][task_id]
                    task_info["completed"] = task_record.get("completed", False)
                    task_info["value"] = task_record.get("value")

                task_status.append(task_info)
        except Exception as e:
            logger.error(f"获取今日任务失败：{e}")
            # 返回空列表或基础任务列表
            task_status = [
                {
                    "task_id": t["task_id"],
                    "task_name": t["task_name"],
                    "task_type": t.get("task_type", "routine"),
                    "completed": False,
                    "value": None,
                    "description": t.get("description", ""),
                    "target_value": t.get("target_value")
                } for t in self.daily_tasks
            ]

        return task_status

    def get_task_summary(self, date=None):
        """获取任务汇总统计"""
        if date is None:
            date = get_date_str()

        try:
            tasks = self.get_today_tasks(date)
            total_tasks = len(tasks)
            completed_tasks = len([t for t in tasks if t["completed"]])

            # 统计必做任务
            required_tasks = [t for t in tasks if next(
                (tt for tt in self.daily_tasks if tt["task_id"] == t["task_id"]),
                {}
            ).get("required", False)]
            completed_required = len([t for t in required_tasks if t["completed"]])

            # 计算完成率
            completion_rate = round(completed_tasks / total_tasks * 100, 2) if total_tasks > 0 else 0
            required_completion_rate = round(
                completed_required / len(required_tasks) * 100, 2
            ) if len(required_tasks) > 0 else 0

            return {
                "date": date,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "completion_rate": completion_rate,
                "required_tasks": len(required_tasks),
                "completed_required": completed_required,
                "required_completion_rate": required_completion_rate
            }
        except Exception as e:
            logger.error(f"获取任务汇总失败：{e}")
            # 返回默认值
            return {
                "date": date,
                "total_tasks": len(self.daily_tasks),
                "completed_tasks": 0,
                "completion_rate": 0.0,
                "required_tasks": 0,
                "completed_required": 0,
                "required_completion_rate": 0.0
            }