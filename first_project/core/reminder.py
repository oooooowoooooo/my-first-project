import json
import schedule
import time
import threading
import os
from datetime import datetime
from plyer import notification
from utils.logger import logger
from utils.time_utils import get_current_time_str, parse_time_str


class TaskReminder:
    def __init__(self, config_path="config/tasks_config.json"):
        # 加载任务配置
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            self.daily_tasks = self.config["daily_tasks"]
        except Exception as e:
            logger.error(f"加载任务配置失败：{e}")
            self.daily_tasks = []

        self.reminder_thread = None
        self.running = False
        self.reminded_tasks = set()  # 避免重复提醒

    def send_notification(self, task_name, message):
        """发送系统通知（跨平台兼容）"""
        try:
            # 构建通知标题和内容
            title = f"📢 任务提醒 | {task_name}"
            # Windows/macOS/Linux 通知适配
            notification.notify(
                title=title,
                message=message,
                app_name="DailyTaskAI",
                timeout=10
            )
            logger.info(f"发送系统提醒：{task_name} - {message}")
        except Exception as e:
            # 降级为控制台输出（所有平台通用）
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n⚠️ 【{timestamp} 任务提醒】{task_name}：{message}")
            logger.warning(f"系统通知发送失败，已降级为控制台提醒：{e}")

    def check_reminders(self):
        """检查当前时间是否有需要提醒的任务"""
        current_time = get_current_time_str()
        current_datetime = datetime.now()
        date_key = current_datetime.strftime("%Y%m%d_%H%M")  # 按分钟去重

        for task in self.daily_tasks:
            task_key = f"{task['task_id']}_{date_key}"
            # 检查是否到提醒时间，且未提醒过
            if current_time in task["reminder_times"] and task_key not in self.reminded_tasks:
                self.reminded_tasks.add(task_key)
                self.send_notification(
                    task["task_name"],
                    f"该完成{task['task_name']}啦！\n{task['description']}"
                )

        # 清理过期的提醒记录（每小时清理一次）
        if current_datetime.minute == 0:
            self.reminded_tasks = set()
            logger.info("清理过期提醒记录")

    def start_scheduler(self):
        """启动提醒调度器（后台运行）"""
        self.running = True
        logger.info("提醒调度器启动")

        # 每分钟检查一次提醒
        schedule.every(1).minutes.do(self.check_reminders)

        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"调度器运行错误：{e}")
                time.sleep(5)  # 出错后延迟5秒再重试

    def start_background(self):
        """在后台线程启动提醒服务"""
        if self.running:
            logger.info("提醒服务已在运行")
            return

        if self.reminder_thread is None or not self.reminder_thread.is_alive():
            self.reminder_thread = threading.Thread(
                target=self.start_scheduler,
                daemon=True,
                name="TaskReminderThread"
            )
            self.reminder_thread.start()
            logger.info("后台提醒服务已启动")

    def stop_reminder(self):
        """停止提醒服务"""
        self.running = False
        # 清空调度器
        schedule.clear()
        # 等待线程结束
        if self.reminder_thread is not None and self.reminder_thread.is_alive():
            self.reminder_thread.join(timeout=3)
        logger.info("提醒服务已停止")