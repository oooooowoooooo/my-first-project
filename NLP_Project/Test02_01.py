import numpy as np


# 基座模型的参数矩阵
W_frozen = np.eye(10)


# 一个 1x10 的向量，全是 1
x = np.ones((1, 10))

# 基座模型的原始输出
y_base = x @ W_frozen

print("基座模型参数量:", W_frozen.size)
print("原始输出:", y_base)
# 1. LoRA 配置
rank = 1
input_dim = 10

# 2. 初始化 LoRA 矩阵
# B 矩阵通常初始化为 0 (Zero Init)，保证刚开始不影响模型
B_lora = np.zeros((rank, input_dim))
# A 矩阵随机初始化 (Gaussian Init)
A_lora = np.random.randn(input_dim, rank)

# 3. 计算 LoRA 增量

delta_W = A_lora @ B_lora

# 4. 融合后的模型输出

y_new = x @ (W_frozen + delta_W)

print("LoRA 参数量:", A_lora.size + B_lora.size) # 参数少
print("挂载后输出:", y_new)
# 目标：我们希望输出全是 11
target = np.ones((1, 10)) * 11

print(f"目标输出: {target[0, 0]}...")
print("-" * 30)


learning_rate = 0.001
losses = []
final_out =[]

for epoch in range(80):  # 多跑几轮
    # 1. Forward
    # path: x(1,10) -> A(10,1) -> B(1,10) -> out(1,10)
    hidden = x @ A_lora  # Shape: (1, 1) - 这是中间隐层
    lora_out = hidden @ B_lora  # Shape: (1, 10)
    final_out = y_base + lora_out

    # 2. Loss
    loss = np.mean((final_out - target) ** 2)

    # 3. Backward

    grad_output = (final_out - target)  # Shape: (1, 10)

    # 3.1求 B 的梯度

    grad_B = hidden.T @ grad_output

    # 3.2 求 A 的梯度
    grad_hidden = grad_output @ B_lora.T  # (1,10) @ (10,1) -> (1,1)


    grad_A = x.T @ grad_hidden

    # 4. Update
    B_lora -= learning_rate * grad_B
    A_lora -= learning_rate * grad_A

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}, Output = {final_out[0, 0]:.2f}")

print("-" * 30)
print("训练完成！")
print("最终输出 (前5位):", final_out[0, :5])

print("LoRA 贡献值:", (final_out - y_base)[0, 0])