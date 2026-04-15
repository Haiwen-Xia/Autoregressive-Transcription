### RoPE-2D Pipeline Formulation

### 1. 任务设定
给定多模态序列输入的注意力张量
$$
\mathbf{X}\in\mathbb{R}^{B\times T\times H\times D},
$$
其中 $B$ 为 batch size，$T$ 为序列长度，$H$ 为头数，$D$ 为每头维度。2D-RoPE 要求
$$
D\equiv 0\pmod 4.
$$

我们希望同时编码：
- 离散序列位置坐标 $p_t=t$
- 连续时间坐标 $\tau_{b,t}$

### 2. 时间坐标构造
设 token 帧率为 $f_{\mathrm{tok}}$，音频帧率为 $f_{\mathrm{aud}}$，时间缩放系数为 $\alpha>0$。

对每个位置 $i$，定义时间坐标 $c_i$：
$$
c_i=
\begin{cases}
\alpha\cdot \dfrac{i_{\mathrm{aud}}}{f_{\mathrm{aud}}}, & \text{audio latent}\\
\alpha\cdot \dfrac{v_i}{f_{\mathrm{tok}}}, & \text{timestamp token (time or time index)}\\
\alpha\cdot s_i, & \text{event attribute with known event time}\\
m_{i-1}+1, & \text{event attribute with unknown event time}\\
m_{i-1}+1, & \text{non-temporal token}
\end{cases}
$$
其中 $s_i$ 是最近一次时间戳解析得到的事件时间，$m_{i-1}=\max_{j<i}c_j$。

对第 $b$ 个样本，记
$$
\boldsymbol{\tau}_b=[\tau_{b,0},\dots,\tau_{b,T-1}]\in\mathbb{R}^{T},\qquad
\tau_{b,t}=c_t.
$$

### 3. 2D-RoPE 的矩阵化表达
令
$$
Q=\frac{D}{4},\qquad
\theta_k=\mathrm{base}^{-k/Q},\quad k=0,\dots,Q-1.
$$

定义位置角与时间角：
$$
\phi^{(p)}_{b,t,k}=p_t\,\theta_k,
\qquad
\phi^{(\tau)}_{b,t,k}=\tau_{b,t}\,\theta_k.
$$

将每个 head 的向量按 4 维分组：
$$
\mathbf{x}_{b,t,h,k}=
\begin{bmatrix}
x_0\\x_1\\y_0\\y_1
\end{bmatrix}
\in\mathbb{R}^{4}.
$$

定义二维块对角旋转矩阵：
$$
\mathbf{R}_{b,t,k}=
\begin{bmatrix}
\cos\phi^{(p)} & -\sin\phi^{(p)} & 0 & 0\\
\sin\phi^{(p)} & \cos\phi^{(p)}  & 0 & 0\\
0 & 0 & \cos\phi^{(\tau)} & -\sin\phi^{(\tau)}\\
0 & 0 & \sin\phi^{(\tau)} & \cos\phi^{(\tau)}
\end{bmatrix}.
$$

则 2D-RoPE 为
$$
\mathbf{x}'_{b,t,h,k}=\mathbf{R}_{b,t,k}\,\mathbf{x}_{b,t,h,k}.
$$

把所有 $k=0,\dots,Q-1$ 的 4 维块拼接回去，即得
$$
\mathbf{X}^{\mathrm{rot}}=\mathrm{RoPE}_{2\mathrm{D}}(\mathbf{X};\mathbf{p},\boldsymbol{\tau}).
$$

### 5. 与注意力结合

在自注意力中，对 $\mathbf{Q},\mathbf{K}$ 施加同一 2D-RoPE：

$$

\tilde{\mathbf{Q}}=\mathrm{RoPE}_{2\mathrm{D}}(\mathbf{Q}),

\qquad

\tilde{\mathbf{K}}=\mathrm{RoPE}_{2\mathrm{D}}(\mathbf{K}).

$$

  

然后计算

$$

\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V})=

\mathrm{softmax}\!\left(\frac{\tilde{\mathbf{Q}}\tilde{\mathbf{K}}^\top}{\sqrt{D}}\right)\mathbf{V}.

$$

  