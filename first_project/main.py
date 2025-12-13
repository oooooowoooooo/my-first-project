import sys
import json
import os
from datetime import datetime
from core.reminder import TaskReminder
from core.task_manager import TaskManager
from core.calorie_tracker import CalorieTracker
from core.reward_system import RewardSystem
from utils.logger import logger
from utils.time_utils import get_date_str


def print_menu():
    """打印主菜单"""
    os.system('cls' if os.name == 'nt' else 'clear')  # 清屏适配
    print("\n===== AI每日行为管理系统 (DailyTaskAI) =====")
    print("1. 启动任务提醒服务（已自动后台运行）")
    print("2. 标记任务完成")
    print("3. 记录卡路里摄入")
    print("4. 查看今日任务完成情况")
    print("5. 查看今日卡路里汇总")
    print("6. 查看积分和等级")
    print("7. 执行每日奖惩结算")
    print("8. 退出系统")
    print("=" * 40)


def input_validation(prompt, input_type="str", allow_empty=False):
    """输入验证工具"""
    while True:
        user_input = input(prompt).strip()
        if not user_input and allow_empty:
            return None
        if not user_input:
            print("❌ 输入不能为空！")
            continue

        if input_type == "int":
            try:
                return int(user_input)
            except ValueError:
                print("❌ 请输入数字！")
        elif input_type == "float":
            try:
                return float(user_input)
            except ValueError:
                print("❌ 请输入数字（可带小数）！")
        else:
            return user_input


def main():
    # 初始化核心模块（确保目录存在）
    for dir_name in ["config", "core", "data", "utils", "logs"]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            logger.info(f"创建目录：{dir_name}")

    # 初始化核心模块
    try:
        reminder = TaskReminder()
        task_manager = TaskManager()
        calorie_tracker = CalorieTracker()
        reward_system = RewardSystem()
        logger.info("核心模块初始化成功")
    except Exception as e:
        print(f"❌ 系统初始化失败：{e}")
        logger.error(f"初始化失败：{e}")
        return

    # 启动提醒服务（后台）
    try:
        reminder.start_background()
        print("✅ 任务提醒服务已在后台启动")
        logger.info("系统启动完成")
    except Exception as e:
        print(f"⚠️ 提醒服务启动失败（不影响核心功能）：{e}")
        logger.warning(f"提醒服务启动失败：{e}")

    while True:
        print_menu()
        choice = input_validation("请输入操作编号（1-8）：", input_type="int")

        if choice == 1:
            print("✅ 任务提醒服务已在后台运行（每分钟检查一次提醒）")

        elif choice == 2:
            # 标记任务完成
            print("\n📋 今日任务列表：")
            tasks = task_manager.get_today_tasks()
            if not tasks:
                print("暂无配置的任务，请检查tasks_config.json")
                input("\n按回车键继续...")
                continue

            for task in tasks:
                status = "✅" if task["completed"] else "❌"
                print(f"{task['task_id']}. {status} {task['task_name']} - {task['description']}")

            task_id = input_validation("请输入要标记完成的任务ID：", input_type="int")
            value_prompt = "请输入任务完成值（如无则回车）："
            value = input(value_prompt).strip()

            if value:
                # 尝试转换为数字（针对数值型任务）
                try:
                    value = float(value)
                except:
                    pass

            if task_manager.mark_task_complete(task_id, value):
                print("✅ 任务标记完成成功！")
            else:
                print("❌ 任务标记完成失败（检查任务ID是否正确）！")

        elif choice == 3:
            # 记录卡路里
            print("\n🍽️  卡路里记录")
            meal_types = {
                1: ("breakfast", "早餐"),
                2: ("lunch", "午餐"),
                3: ("dinner", "晚餐")
            }
            for key, (_, name) in meal_types.items():
                print(f"{key}. {name}")

            meal_choice = input_validation("请选择餐次（1-3）：", input_type="int")
            if meal_choice not in meal_types:
                print("❌ 无效选择（仅支持1-3）！")
                input("\n按回车键继续...")
                continue

            calories = input_validation("请输入卡路里数值（正数）：", input_type="float")
            if calories < 0:
                print("❌ 卡路里不能为负数！")
                input("\n按回车键继续...")
                continue

            meal_key, _ = meal_types[meal_choice]
            if calorie_tracker.record_calorie(meal_key, calories):
                print("✅ 卡路里记录成功！")
                # 即时提示是否超标
                summary = calorie_tracker.get_daily_calorie_summary()
                if not summary["within_limit"]:
                    print(f"⚠️ 警告：当前总卡路里{summary['total']}已超过每日限制{summary['limits']['daily_total']}！")
            else:
                print("❌ 卡路里记录失败！")

        elif choice == 4:
            # 查看今日任务
            print(f"\n📊 【{get_date_str()}】任务完成情况")
            summary = task_manager.get_task_summary()
            print(f"├─ 总任务数：{summary['total_tasks']} | 已完成：{summary['completed_tasks']}")
            print(f"├─ 完成率：{summary['completion_rate']}%")
            print(f"└─ 必做任务：{summary['completed_required']}/{summary['required_tasks']}")

            tasks = task_manager.get_today_tasks()
            print("\n详细任务：")
            for task in tasks:
                status = "✅" if task["completed"] else "❌"
                value = f" (值：{task['value']})" if task['value'] is not None else ""
                print(f"  {status} {task['task_name']}{value}")

        elif choice == 5:
            # 查看卡路里
            print(f"\n🍽️  【{get_date_str()}】卡路里汇总")
            summary = calorie_tracker.get_daily_calorie_summary()
            print(f"├─ 今日总摄入：{summary['total']} 卡路里")
            print(f"├─ 每日限制：{summary['limits']['daily_total']} 卡路里")
            print(f"└─ 是否超标：{'❌ 是' if not summary['within_limit'] else '✅ 否'}")

            print("\n各餐次详情：")
            meal_mapping = {
                "breakfast": "早餐",
                "lunch": "午餐",
                "dinner": "晚餐"
            }
            for meal_type, calories in summary["meals"].items():
                limit = summary["limits"][meal_type]
                status = "❌ 超标" if calories > limit else "✅ 正常"
                print(f"  {meal_mapping.get(meal_type, meal_type)}：{calories} / {limit} {status}")

        elif choice == 6:
            # 查看积分和等级
            print("\n🏆 积分与等级系统")
            level_info = reward_system.get_user_level()
            print(f"├─ 当前总积分：{level_info['total_points']}")
            print(f"├─ 当前等级：{level_info['level']}级 - {level_info['name']}")
            print(f"└─ 下一等级：{level_info['next_level']}")

            # 显示今日积分
            try:
                with open("data/task_records.json", "r", encoding="utf-8") as f:
                    records = json.load(f)
                today_points = records.get(get_date_str(), {}).get("points", 0)
                print(f"\n今日积分变动：{today_points}")
            except:
                print("\n今日积分变动：0（暂无记录）")

        elif choice == 7:
            # 执行奖惩结算
            print("\n⚖️  执行每日奖惩结算...")
            try:
                reward_system.check_daily_rewards()
                print("✅ 奖惩结算完成！")
                # 结算后显示最新等级
                level_info = reward_system.get_user_level()
                print(f"当前等级：{level_info['name']} (积分：{level_info['total_points']})")
            except Exception as e:
                print(f"❌ 结算失败：{e}")
                logger.error(f"奖惩结算失败：{e}")

        elif choice == 8:
            # 退出系统
            print("\n👋 正在退出系统...")
            try:
                reminder.stop_reminder()
            except:
                pass
            logger.info("系统正常退出")
            print("✅ 系统已安全退出，再见！")
            sys.exit(0)

        else:
            print("❌ 无效的选择（仅支持1-8）！")

        input("\n按回车键继续...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断操作，系统退出")
        logger.info("用户中断程序")
    except Exception as e:
        print(f"\n❌ 系统运行出错：{e}")
        logger.error(f"系统崩溃：{e}", exc_info=True)