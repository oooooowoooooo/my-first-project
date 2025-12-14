import streamlit as st
import json
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

# 配置页面
st.set_page_config(
    page_title="AI每日行为管理系统",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入核心模块
from core.task_manager import TaskManager
from core.calorie_tracker import CalorieTracker
from core.reward_system import RewardSystem
from core.reminder import TaskReminder
from utils.time_utils import get_date_str, get_past_n_days

# 初始化核心模块
task_manager = TaskManager()
calorie_tracker = CalorieTracker()
reward_system = RewardSystem()
reminder = TaskReminder()

# 侧边栏
st.sidebar.title("📋 AI行为管理系统")
page = st.sidebar.radio(
    "功能菜单",
    ["仪表盘", "任务管理", "卡路里记录", "积分等级", "历史统计", "系统设置"]
)

# 全局样式
st.markdown("""
<style>
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .success {
        background-color: #f0f8fb;
        border-left: 4px solid #28a745;
    }
    .warning {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
    }
    .danger {
        background-color: #fef7fb;
        border-left: 4px solid #dc3545;
    }
    .stat-card {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 1. 仪表盘页面（首页）
if page == "仪表盘":
    st.title("📊 每日行为仪表盘")
    st.divider()

    # 今日日期
    today = get_date_str()
    st.subheader(f"今日进度 ({today})")

    # 统计卡片行
    col1, col2, col3, col4 = st.columns(4)

    # 任务完成率
    task_summary = task_manager.get_task_summary()
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h4>任务完成率</h4>
            <h2>{task_summary['completion_rate']}%</h2>
            <p>{task_summary['completed_tasks']}/{task_summary['total_tasks']} 任务</p>
        </div>
        """, unsafe_allow_html=True)

    # 必做任务完成
    with col2:
        status = "success" if task_summary['completed_required'] == task_summary['required_tasks'] else "warning"
        st.markdown(f"""
        <div class="stat-card">
            <h4>必做任务</h4>
            <h2>{task_summary['completed_required']}/{task_summary['required_tasks']}</h2>
            <p>完成率 {task_summary['required_completion_rate']}%</p>
        </div>
        """, unsafe_allow_html=True)

    # 卡路里情况
    calorie_summary = calorie_tracker.get_daily_calorie_summary()
    with col3:
        limit_status = "success" if calorie_summary['within_limit'] else "danger"
        st.markdown(f"""
        <div class="stat-card">
            <h4>卡路里摄入</h4>
            <h2>{calorie_summary['total']}/{calorie_summary['limits']['daily_total']}</h2>
            <p>{"✅ 达标" if calorie_summary['within_limit'] else "❌ 超标"}</p>
        </div>
        """, unsafe_allow_html=True)

    # 用户等级
    level_info = reward_system.get_user_level()
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <h4>当前等级</h4>
            <h2>{level_info['name']}</h2>
            <p>积分: {level_info['total_points']} | 连续: {level_info['streak_days']}天</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 今日任务列表
    st.subheader("📋 今日任务")
    tasks = task_manager.get_today_tasks()

    # 分栏显示完成/未完成任务
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### ✅ 已完成任务")
        completed_tasks = [t for t in tasks if t['completed']]
        if completed_tasks:
            for task in completed_tasks:
                st.markdown(f"""
                <div class="card success">
                    <h5>{task['task_name']}</h5>
                    <p>完成时间: {task['completed_at']}</p>
                    {f"<p>完成值: {task['value']}</p>" if task['value'] else ""}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("暂无已完成任务")

    with col_right:
        st.markdown("### ❌ 未完成任务")
        uncompleted_tasks = [t for t in tasks if not t['completed']]
        if uncompleted_tasks:
            for task in uncompleted_tasks:
                st.markdown(f"""
                <div class="card warning">
                    <h5>{task['task_name']}</h5>
                    <p>{task['description']}</p>
                    {f"<p>目标值: {task['target_value']}</p>" if task['target_value'] else ""}
                    <p>提醒时间: {', '.join(task['reminder_times'])}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 所有任务都已完成！")

    st.divider()

    # 今日积分变动
    st.subheader("🏆 今日积分变动")
    daily_points = reward_system.get_daily_points()

    if daily_points["points_change"]:
        df_points = pd.DataFrame(daily_points["points_change"])
        st.dataframe(
            df_points,
            column_config={
                "time": st.column_config.TextColumn("时间"),
                "points": st.column_config.NumberColumn("积分变动", format="%+d"),
                "reason": st.column_config.TextColumn("原因")
            },
            hide_index=True
        )
        st.markdown(f"**今日总积分变动**: {daily_points['daily_points']:+d}")
    else:
        st.info("今日暂无积分变动")

# 2. 任务管理页面
elif page == "任务管理":
    st.title("📋 任务管理")
    st.divider()

    # 标记任务完成
    st.subheader("✅ 标记任务完成")
    tasks = task_manager.get_today_tasks()

    # 任务选择
    task_options = {f"{t['task_id']} - {t['task_name']}": t for t in tasks if not t['completed']}
    if task_options:
        selected_task_str = st.selectbox("选择要完成的任务", list(task_options.keys()))
        selected_task = task_options[selected_task_str]

        # 完成值输入
        task_value = st.text_input(
            "完成值（如学习时长/博客数量，无则留空）",
            placeholder=f"目标值: {selected_task['target_value']}" if selected_task['target_value'] else ""
        )

        # 提交按钮
        if st.button("标记为完成", type="primary"):
            if task_manager.mark_task_complete(selected_task['task_id'], task_value):
                st.success(f"✅ 成功标记「{selected_task['task_name']}」为完成！")
                st.rerun()
            else:
                st.error("❌ 标记失败，请重试！")
    else:
        st.success("🎉 所有任务都已完成！")

    st.divider()

    # 任务列表
    st.subheader("📜 所有任务列表")
    all_tasks = task_manager.get_today_tasks()

    # 转换为DataFrame显示
    df_tasks = pd.DataFrame(all_tasks)
    df_tasks['状态'] = df_tasks['completed'].apply(lambda x: "已完成" if x else "未完成")
    df_tasks = df_tasks[['task_id', 'task_name', '状态', 'value', 'target_value', 'description']]

    st.dataframe(
        df_tasks,
        column_config={
            "task_id": "任务ID",
            "task_name": "任务名称",
            "状态": "状态",
            "value": "完成值",
            "target_value": "目标值",
            "description": "描述"
        },
        hide_index=True,
        use_container_width=True
    )

# 3. 卡路里记录页面
elif page == "卡路里记录":
    st.title("🍽️ 卡路里记录")
    st.divider()

    # 记录卡路里
    st.subheader("📝 记录餐次卡路里")
    col1, col2 = st.columns(2)

    with col1:
        meal_type = st.selectbox(
            "选择餐次",
            [("早餐", "breakfast"), ("午餐", "lunch"), ("晚餐", "dinner")],
            format_func=lambda x: x[0]
        )

    with col2:
        calories = st.number_input(
            "卡路里数值",
            min_value=0.0,
            step=10.0,
            help="请输入该餐摄入的卡路里数值"
        )

    if st.button("保存记录", type="primary"):
        if calorie_tracker.record_calorie(meal_type[1], calories):
            st.success(f"✅ 成功记录{meal_type[0]}卡路里：{calories}！")
            st.rerun()
        else:
            st.error("❌ 记录失败，请重试！")

    st.divider()

    # 今日卡路里汇总
    st.subheader("📊 今日卡路里汇总")
    calorie_summary = calorie_tracker.get_daily_calorie_summary()

    # 可视化
    meal_names = {"breakfast": "早餐", "lunch": "午餐", "dinner": "晚餐"}
    meal_data = []

    for meal_type, data in calorie_summary["meals"].items():
        meal_data.append({
            "餐次": meal_names.get(meal_type, meal_type),
            "摄入": data["value"],
            "限制": data["limit"],
            "超标": data["over_limit"]
        })

    if meal_data:
        df_calorie = pd.DataFrame(meal_data)

        # 柱状图
        fig = px.bar(
            df_calorie,
            x="餐次",
            y=["摄入", "限制"],
            barmode="group",
            title="各餐次卡路里对比",
            color_discrete_map={"摄入": "#ff6b6b", "限制": "#4ecdc4"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # 详情表格
        st.dataframe(
            df_calorie,
            column_config={
                "餐次": "餐次",
                "摄入": st.column_config.NumberColumn("摄入卡路里"),
                "限制": st.column_config.NumberColumn("限制卡路里"),
                "超标": st.column_config.CheckboxColumn("是否超标")
            },
            hide_index=True
        )
    else:
        st.info("暂无卡路里记录")

    # 总计信息
    st.markdown(f"""
    <div class="card {'danger' if not calorie_summary['within_limit'] else 'success'}">
        <h4>今日总计</h4>
        <p>总摄入：{calorie_summary['total']} 卡路里</p>
        <p>每日限制：{calorie_summary['limits']['daily_total']} 卡路里</p>
        <p>状态：{"❌ 超标" if not calorie_summary['within_limit'] else "✅ 达标"}</p>
    </div>
    """, unsafe_allow_html=True)

# 4. 积分等级页面
elif page == "积分等级":
    st.title("🏆 积分与等级系统")
    st.divider()

    # 用户当前信息
    level_info = reward_system.get_user_level()
    st.subheader("📈 个人信息")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="card">
            <h4>当前等级：{level_info['level']}级 - {level_info['name']}</h4>
            <p>总积分：{level_info['total_points']}</p>
            <p>连续完成天数：{level_info['streak_days']} 天</p>
            <p>下一等级：{level_info['next_level']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # 等级体系展示
        st.markdown("### 等级体系")
        reward_config = json.load(open("config/rewards_config.json", "r", encoding="utf-8"))
        level_system = reward_config["level_system"]

        level_data = []
        for level_id, info in level_system.items():
            level_data.append({
                "等级": f"{level_id}级",
                "名称": info["name"],
                "积分范围": f"{info['min_points']} - {info['max_points']}"
            })

        st.dataframe(
            pd.DataFrame(level_data),
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    # 今日积分变动
    st.subheader("📝 今日积分变动")
    daily_points = reward_system.get_daily_points()

    if daily_points["points_change"]:
        # 积分变动列表
        st.dataframe(
            pd.DataFrame(daily_points["points_change"]),
            column_config={
                "time": "时间",
                "points": st.column_config.NumberColumn("积分变动", format="%+d"),
                "reason": "变动原因"
            },
            hide_index=True,
            use_container_width=True
        )

        # 今日总计
        st.markdown(f"""
        <div class="card">
            <h4>今日积分总计：{daily_points['daily_points']:+d}</h4>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("今日暂无积分变动")

    st.divider()

    # 奖惩规则
    st.subheader("📜 奖惩规则")
    reward_config = json.load(open("config/rewards_config.json", "r", encoding="utf-8"))

    tab1, tab2 = st.tabs(["奖励规则", "惩罚规则"])

    with tab1:
        rewards = reward_config["rewards"]
        for reward_name, reward_info in rewards.items():
            st.markdown(f"""
            <div class="card success">
                <h5>{reward_info['message']}</h5>
                <p>积分变动：+{reward_info['points']}</p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        punishments = reward_config["punishments"]
        for punish_name, punish_info in punishments.items():
            points = punish_info['points']
            st.markdown(f"""
            <div class="card danger">
                <h5>{punish_info['message']}</h5>
                <p>积分变动：{points}</p>
                {f"<p>触发阈值：{punish_info['threshold']}个任务</p>" if 'threshold' in punish_info else ""}
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # 手动结算按钮
    if st.button("🔄 执行今日奖惩结算", type="primary"):
        reward_system.check_daily_rewards()
        st.success("✅ 奖惩结算完成！")
        st.rerun()

# 5. 历史统计页面
elif page == "历史统计":
    st.title("📈 历史统计")
    st.divider()

    # 选择统计天数
    days = st.slider("选择统计天数", min_value=1, max_value=30, value=7)
    past_days = get_past_n_days(days)

    # 任务完成趋势
    st.subheader("📋 任务完成趋势")
    task_history = task_manager.get_task_history(days)

    if task_history:
        df_task = pd.DataFrame(task_history)
        df_task['日期'] = df_task['date']
        df_task['完成率'] = df_task['completion_rate']

        # 折线图
        fig1 = px.line(
            df_task,
            x="日期",
            y="完成率",
            title=f"{days}天任务完成率趋势",
            markers=True,
            range_y=[0, 100]
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 数据表格
        st.dataframe(
            df_task[['日期', 'total_tasks', 'completed_tasks', 'completion_rate']],
            column_config={
                "日期": "日期",
                "total_tasks": "总任务数",
                "completed_tasks": "已完成数",
                "completion_rate": st.column_config.NumberColumn("完成率(%)", format="%.1f")
            },
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    # 卡路里趋势
    st.subheader("🍽️ 卡路里摄入趋势")
    calorie_history = []

    for day in past_days:
        summary = calorie_tracker.get_daily_calorie_summary(day)
        calorie_history.append({
            "日期": day,
            "总摄入": summary['total'],
            "限制": summary['limits']['daily_total'],
            "是否达标": summary['within_limit']
        })

    if calorie_history:
        df_calorie = pd.DataFrame(calorie_history)

        # 柱状图
        fig2 = px.bar(
            df_calorie,
            x="日期",
            y=["总摄入", "限制"],
            barmode="group",
            title=f"{days}天卡路里摄入趋势",
            color_discrete_map={"总摄入": "#ff6b6b", "限制": "#4ecdc4"}
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 数据表格
        st.dataframe(
            df_calorie,
            column_config={
                "日期": "日期",
                "总摄入": "总摄入(卡路里)",
                "限制": "每日限制(卡路里)",
                "是否达标": st.column_config.CheckboxColumn("是否达标")
            },
            hide_index=True,
            use_container_width=True
        )

# 6. 系统设置页面
elif page == "系统设置":
    st.title("⚙️ 系统设置")
    st.divider()

    # 提醒服务控制
    st.subheader("🔔 提醒服务")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("启动提醒服务", type="primary"):
            reminder.start_background()
            st.success("✅ 提醒服务已启动！")

    with col2:
        if st.button("停止提醒服务"):
            reminder.stop_reminder()
            st.success("✅ 提醒服务已停止！")

    st.divider()

    # 配置文件展示
    st.subheader("📝 配置文件")

    tab1, tab2 = st.tabs(["任务配置", "奖惩配置"])

    with tab1:
        task_config = json.load(open("config/tasks_config.json", "r", encoding="utf-8"))
        st.json(task_config)

    with tab2:
        reward_config = json.load(open("config/rewards_config.json", "r", encoding="utf-8"))
        st.json(reward_config)

    st.divider()

    # 数据管理
    st.subheader("🗂️ 数据管理")

    if st.button("📥 导出数据", type="secondary"):
        with open("data/task_records.json", "r", encoding="utf-8") as f:
            data = f.read()

        st.download_button(
            label="下载数据文件",
            data=data,
            file_name=f"task_records_{get_date_str()}.json",
            mime="application/json"
        )

    if st.button("⚠️ 清空今日数据", type="secondary"):
        if st.checkbox("确认清空今日数据"):
            records = reward_system._get_records()
            today = get_date_str()
            if today in records["daily_records"]:
                del records["daily_records"][today]
                reward_system._save_records(records)
                st.success("✅ 今日数据已清空！")
                st.rerun()