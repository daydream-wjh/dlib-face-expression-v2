# Dlib 2.0

基于 **Dlib** 的人脸表情识别 2.0：沿用 1.0 的"68 关键点 → 几何特征"路线，把特征整理成表格数据集，并补齐**多分类器横向对比**（SVM / KNN / 决策树 / 逻辑回归）与 **PSO 粒子群自动调参**，训练好的模型用 `joblib` 落盘，实时识别端加载 KNN 模型上屏。

> 1.0 是 2019 年四人小组课程项目（已申请软件著作权），本仓库是其后续重构版本；改造前的原版见 [`dlib-face-expression`](https://github.com/daydream-wjh/dlib-face-expression) 仓库。

## 项目简介

1.0 的做法是把"摄像头抓帧 → 算特征 → KNN 分类"全部写死在三个文件里，模型只有一个来路不明的 `.m` 文件，没有数据集也没有训练脚本。2.0 把这条链路拆开：

- **数据侧**：`face/data.py` 负责对单张人脸图提取 6 个几何特征，`数据集/*.xlsx` 存放整理好的特征表（6 个特征列 + 1 个标签列）；
- **训练侧**：`face/SVM.py`、`face/KNN.py`、`face/DecisionTreeClassifier.py`、`face/LogisticRegression.py` 各读一份 xlsx、交叉验证、训练、`joblib.dump` 保存模型；`face/pso.py` 用 pyswarms 的全局最优 PSO 搜 `SVC(C, gamma)` 的最优组合；
- **推理侧**：`face/main.py` → `get_detector.py` → `learning_face.py` → `face_pd.py`，与 1.0 相同的实时链路，只是模型换成 `knn_model.pkl`。

## 数据集

仓库内 `数据集/` 是**特征表**而不是原始图片：每行是一个人脸样本，前 6 列是 `learning_face.py` 里那套几何特征，最后 1 列是表情标签。

`数据集/说明.txt`（GBK 编码，`cat` 出来是乱码，需 `iconv -f GBK -t UTF-8`）给出了列序与标签定义：

```
第一列.mouth_width      第二列.mouth_higth    第三列.brow_hight
第四列.brow_width       第五列.eye_hight      第六列.brow_k
第七列.标签（1.愤怒 2.蔑视 3.恶心 4.害怕 5.开心 6.悲伤 7.惊喜）
```

同文件还记录了特征提取时的预处理：`cv2.resize(im_rd, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)`，以及 `face_hight = face_width`。（`face/data.py` 里实际写的是 `fx=2, fy=2`，与说明不一致；`learning_face.py` 则完全不 resize，直接用摄像头原帧。）

各文件实际内容（读取核对过行列数与标签分布）：

| 文件 | 规模 | 列结构 | 数值区间 | 标签分布 |
| --- | --- | --- | --- | --- |
| `数据.xlsx` | 728 行 | 7 列有值（6 特征 + 标签），第 8–9 列为空 | 比例值，第 1 列 0.218–0.606 | `1:135, 2:53, 5:207, 6:84, 7:249`（**只有 5 类，缺 3/4**） |
| `train_dataset.xlsx` | 20352 行 | 6 列（5 特征 + 标签） | 比例值，0.139–0.642 | `1:2817, 2:341, 3:2583, 4:5689, 5:3749, 6:2769, 7:2404` |
| `test_dataset.xlsx` | 5074 行 | 6 列（5 特征 + 标签） | 比例值，0.155–0.663 | `1:665, 2:85, 3:658, 4:1401, 5:944, 6:705, 7:616` |
| `train_dataset2.xlsx` | 20352 行 | 7 列（6 特征 + 标签） | **原始像素量级**，第 1 列 −71–184.5 | 同 `train_dataset.xlsx` |
| `test_dataset2.xlsx` | 5074 行 | 7 列（6 特征 + 标签） | 原始像素量级，−46–155 | 同 `test_dataset.xlsx` |
| `train_dataset3.xlsx` | 982 行 | 7 列（6 特征 + 标签） | 原始像素量级，−11.5–77 | `1:135, 2:54, 3:177, 4:75, 5:207, 6:84, 7:250`（7 类齐全） |

也就是说存在两套编码：`train/test_dataset*.xlsx` 是**大批量原始量纲**（`*2` 版本多一列 `brow_k`），`数据.xlsx` 是**比例归一化后的小样本**（728 行，5 类）。脚本各取所需，因此实验之间不完全可比，详见"已知问题"。

`数据集/S005_001_00000009.png` 是一张 48×48 的灰度人脸裁剪图（`data.py` 里硬编码的那张样例输入，文件名是 `S005_001_00000009` 这种"受试者_表情_帧号"的命名风格）。

## 模型 / 算法

### 特征（与 1.0 相同）

基于 Dlib 68 点关键点、以人脸框宽 `face_width` 归一化的几何量：

| 变量 | 含义 | 计算方式 |
| --- | --- | --- |
| `a` | 嘴巴咧开程度 | `(p54.x - p48.x) / face_width` |
| `b` | 嘴巴张开程度 | `(p66.y - p62.y) / face_width` |
| `c` | 眉毛高度占比 | 20 个眉毛点相对人脸框顶部的平均偏移 / `face_width` |
| `dd` | 眉毛间距占比 | `(Σ_{j=17..20}(p_{j+5}.x - p_j.x) / 5) / face_width` |
| `e` | 眼睛睁开程度 | `((p41.y-p37.y)+(p40.y-p38.y)+(p47.y-p43.y)+(p46.y-p44.y)) / 4 / face_width` |
| `f` | 眉毛倾斜斜率 | 眉毛点 `np.polyfit(x, y, 1)` 的斜率取反（`self.brow_k`） |

实时推理端（`learning_face.py`）只把 `a, b, c, dd, e` 传给分类器，`f` 算了但没用；训练脚本里 SVM / 决策树读 6 列特征（含 `f`），KNN / 逻辑回归只读 5 列。

### 四个分类器（各自脚本的配置）

| 脚本 | 读取数据 | 特征列 | 模型与超参 | 验证方式 | 输出文件 |
| --- | --- | --- | --- | --- | --- |
| `SVM.py` | `test_dataset2.xlsx` | 第 0–5 列（6 个） | `SVC(kernel='poly')`，特征先 `preprocessing.scale` | `cross_val_score(cv=10)` 十折 | `svm_model.pkl` |
| `KNN.py` | `数据.xlsx` | 第 0–4 列（5 个，**跳过 `brow_k`**） | `KNeighborsClassifier` + `GridSearchCV(n_neighbors=[3,5,…,19], cv=10)` | `train_test_split(test_size=0.2, random_state=0)` | `knn_model.pkl` |
| `DecisionTreeClassifier.py` | `数据.xlsx` | 第 0–5 列（6 个） | `DecisionTreeClassifier()` 全默认参数 | `cross_val_score(cv=10)` | `dtc_model.pkl` |
| `LogisticRegression.py` | `数据.xlsx` | `usecols="A:E"`（5 个） | `LogisticRegression(multi_class='ovr', solver='liblinear')`，`StandardScaler` 标准化 | `cross_val_score(cv=10)` | `svm_model.pkl`（**与 SVM 同名，会覆盖**） |

### PSO 调参（`pso.py`）

- 目标函数：给定 `(C, gamma)` 训练 `SVC(kernel='rbf')`，返回 **`1 - 测试集准确率`**（最小化错误率）；
- **逐粒子评估**：PySwarms 一次性把整个粒子矩阵（`shape=(n_particles, dimensions)`）传给目标函数，所以函数内部 `np.atleast_2d` 归一化后逐行遍历、对每个粒子拟合一次 SVC，最后返回长度等于粒子数的**代价数组**（先前 `C = params[0]` 取到的是「第一个粒子的参数向量」，`SVC(C=array)` 必崩，已修正；同时兼容传入单个 1-D 参数向量的调用方式）；
- 数据集：`数据.xlsx` 的第 0–4 列特征 + 第 6 列标签，`train_test_split(test_size=0.2, random_state=42)`；数据和切分都在目标函数外只做一次；
- 粒子群：`pyswarms.single.GlobalBestPSO(n_particles=10, dimensions=2, options={'c1':0.5,'c2':0.3,'w':0.9}, bounds=([0.1,0.1],[100,100]))`，`iters=100`；
- 搜索空间：`C ∈ [0.1, 100]`、`gamma ∈ [0.1, 100]`；
- 结果：`optimizer.optimize()` 返回的是 **`(best_cost, best_pos)` 元组**（旧代码直接把整个元组赋给 `c`，再用未定义的 `g` 打印，末尾必 `NameError`），现在解包后打印一行：
  `最佳参数: C=<C>, gamma=<gamma>, 最优得分=<best_cost>`
  其中最优得分就是**最低的分类错误率**（`1 - 准确率`），不是准确率。

### 评估指标

脚本只 `print` 十折交叉验证的均值与训练集自评分的准确率，**没有任何数字被写入仓库**（无日志、无 README 记录、无图表），因此这里不给具体准确率，跑一遍脚本即可在终端看到：

```
十倍交叉验证的准确率为 <acc>
KNN：在训练集上分类准确率为 <best_score_> / 最好的超参数为 <best_params> / 在测试集上的分类准确率为 <acc>
```

`pso.py` 只在最后打印一行（见上节），`<C>` / `<gamma>` 保留 4 位小数：

```
最佳参数: C=<C>, gamma=<gamma>, 最优得分=<best_cost>
```

### 仓库内已有模型文件（核对结果）

| 文件 | 大小 | 实际内容 |
| --- | --- | --- |
| `knn_model.pkl` | 64 KB | `GridSearchCV` 包裹的 `KNeighborsClassifier`（含 `sklearn.model_selection._search`、`neighbors._classification`、`_kd_tree`） |
| `dtc_model.pkl` | 4.1 KB | `DecisionTreeClassifier`（`sklearn.tree._classes` + `_tree`） |
| `svm_model.pkl` | 1.1 KB | 里面是 **`sklearn.linear_model._logistic`（逻辑回归）**，不是 SVM——被 `LogisticRegression.py` 的同名 `joblib.dump` 覆盖了 |

## 2.0 相对 1.0 增加了什么

| 方面 | 1.0 | 2.0 |
| --- | --- | --- |
| 数据集 | 无（模型来路不明） | 新增 `数据集/`，6 份 xlsx 特征表 + `说明.txt` 列定义 |
| 特征提取脚本 | 无 | 新增 `face/data.py`：对单张图提取特征并尝试写入 xlsx（草稿状态） |
| 模型 | 仅一个 KNN `.m` 文件，缺失 | 4 个分类器脚本 + 3 个 `.pkl`，可复现训练 |
| 分类器数量 | 1（KNN） | 4（SVM / KNN / 决策树 / 逻辑回归）+ 横向对比 |
| 超参搜索 | 无 | KNN 网格搜索（`n_neighbors` 3–19，10 折）+ **PSO 搜 SVM 的 `C`/`gamma`**（10 粒子、100 次迭代） |
| 疲劳检测分支 | 有（`e<0.02` 打 `"tired"`，还 `import music`） | **已移除**，`face_pd.py` 只做表情分类，不再依赖 `music` 模块 |
| 类/入口命名 | `face_emotion` 类 + `face_emotion.py` 调度文件 | 统一为 `faceEmotion` 类，`main.py` 直接 import 三个模块（`face_emotion.py` 已删除） |
| 模型路径 | 硬编码中文文件名 `新最后最后最后五围2knn.m` | 改为 `knn_model.pkl` |
| 摄像头宽度 | `cap.set(3, 480)` | `cap.set(3, 1024)` |

## 技术栈与依赖

按代码 import 推断：

- `dlib`（`get_frontal_face_detector` + `shape_predictor`）
- `opencv-python`（视频流、绘制、`resize`、`imread`）
- `numpy`（`np.polyfit`、数组运算）
- `pandas`（`read_excel`，读 xlsx 需 `openpyxl` 引擎）
- `scikit-learn`（`SVC`、`KNeighborsClassifier`、`GridSearchCV`、`DecisionTreeClassifier`、`LogisticRegression`、`preprocessing`、`cross_val_score`、`train_test_split`）
- `joblib`（模型持久化）
- `xlwt`（`data.py` 里写 xls，Python 2/3 通用的老库）
- `pyswarms`（`pso.py` 的 PSO 实现）

```bash
pip install dlib opencv-python numpy pandas openpyxl scikit-learn joblib xlwt pyswarms
```

无 `requirements.txt`；`.pkl` 由 2019 年前后的 scikit-learn 版本生成，新版本 `joblib.load` 时可能出现 `InconsistentVersionWarning`。

## 目录结构

```
dlib-face-expression-v2/
├── README.md
├── .gitignore
├── .idea/                                     # PyCharm 工程文件
├── face/
│   ├── main.py                                # 入口：实例化 faceEmotion，依次调 get_detector / learning_face
│   ├── get_detector.py                        # Dlib 检测器 + 68 点模型 + 摄像头初始化
│   ├── learning_face.py                       # 实时主循环：抓帧、检测、画 68 点、算 a/b/c/dd/e
│   ├── face_pd.py                             # 加载 knn_model.pkl，预测并把表情英文名画到人脸框下
│   ├── data.py                                # 单张图提特征 → 写 xlsx 的草稿脚本（未跑通）
│   ├── SVM.py                                 # SVC(kernel='poly') + 十折交叉验证 → svm_model.pkl
│   ├── KNN.py                                 # GridSearchCV 调 n_neighbors → knn_model.pkl
│   ├── DecisionTreeClassifier.py              # 默认参数决策树 → dtc_model.pkl
│   ├── LogisticRegression.py                  # ovr/liblinear 逻辑回归 → svm_model.pkl（覆盖）
│   ├── pso.py                                 # PSO 搜索 SVC 的 C / gamma
│   ├── knn_model.pkl                          # GridSearchCV(KNN)，推理端实际使用
│   ├── svm_model.pkl                          # 实为逻辑回归（被覆盖后留下）
│   ├── dtc_model.pkl                          # 决策树
│   └── shape_predictor_68_face_landmarks.dat   # Dlib 68 点模型（约 96 MB）
└── 数据集/
    ├── 说明.txt                                # 列序 + 标签含义（GBK 编码）
    ├── S005_001_00000009.png                  # 特征提取样例图（48×48 灰度）
    ├── 数据.xlsx                               # 728 行，5 特征/6 特征 + 标签，仅 5 类
    ├── train_dataset.xlsx / test_dataset.xlsx  # 20352 / 5074 行，5 特征 + 标签（比例值）
    ├── train_dataset2.xlsx / test_dataset2.xlsx# 20352 / 5074 行，6 特征 + 标签（原始量级）
    └── train_dataset3.xlsx                     # 982 行，6 特征 + 标签，7 类齐全
```

### `face/` 每个脚本各管什么

- **`main.py`**：唯一入口，`import get_detector, learning_face, data`（`data` 导入但未使用），实例化 `faceEmotion`，调用 `get_detector()` 建检测器/摄像头，再调用 `learning_face()` 进实时循环。
- **`get_detector.py`**：`dlib.get_frontal_face_detector()`、`dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")`、`cv2.VideoCapture(0)`、`cap.set(3, 144)`、截图计数器 `cnt = 0`。
- **`learning_face.py`**：真正的识别循环，负责画框/画点/连线、算 5 个特征、调用 `face_pd.face_pd()`；处理 `S` 截图与 `ESC` 退出。
- **`face_pd.py`**：模型推理末梢，`joblib.load("knn_model.pkl")` → `predict([[a,b,c,dd,e]])` → `emo[kll-1]` 上屏（`emo = ['angry','sadness','nature','nature','happy','normal','surprise']`）。
- **`data.py`**：把 `learning_face.py` 的特征计算复制到"单张图 → 一行特征"的场景，想用 `xlwt` 建 `mysheet` 写成训练表；实际未完成——路径是 Windows 绝对路径，`sheet.write(...)` 被注释，`book.save()` 没传文件名。属于留档草稿。
- **`SVM.py` / `KNN.py` / `DecisionTreeClassifier.py` / `LogisticRegression.py`**：四个独立训练脚本，互不 import，各自读 xlsx、评估、`joblib.dump`，可单独运行对比。
- **`pso.py`**：独立调参脚本，把 `SVC` 的 `C`/`gamma` 交给 pyswarms 优化，不保存模型，只打印最优参数（`最佳参数: C=..., gamma=..., 最优得分=...`）；不再 `from sklearn.datasets import load_iris`（导入了但从未使用，已删除）。

## 快速开始

### 1. 实时表情识别

```bash
cd face
python main.py
```

需要：可用的摄像头、`shape_predictor_68_face_landmarks.dat` 与 `knn_model.pkl` 都在 `face/` 下（代码用相对工作目录的路径）。运行后画面里出现红框 + 68 点连线的表情标注，`S` 截图、`ESC` 退出。

### 2. 重新训练各个分类器

**必须在 `face/` 目录下运行**——4 个训练脚本读的都是 `../数据集/xxx.xlsx`：

```bash
cd face
python KNN.py                     # 打印最优 n_neighbors、训练/测试准确率，存 knn_model.pkl
python SVM.py                     # 打印十折 CV 与训练集自评分，存 svm_model.pkl
python DecisionTreeClassifier.py  # 存 dtc_model.pkl
python LogisticRegression.py      # 注意：也会写 svm_model.pkl
```

若不想让逻辑回归覆盖 SVM 模型，先改掉 `LogisticRegression.py` 末尾的 `joblib.dump(clf, "svm_model.pkl")`。

### 3. PSO 调参

```bash
cd face
python pso.py
```

跑 100 次迭代 × 10 个粒子（每次迭代里每个粒子都要拟合一个 RBF-SVC），耗时取决于机器；结束时打印一行最优组合：

```
最佳参数: C=<C>, gamma=<gamma>, 最优得分=<best_cost>
```

`best_cost` 是**最低分类错误率**（`1 - 测试集准确率`），例如 `0.0450` 表示测试集准确率约 95.5%。脚本不落盘模型，要用这组参数得自己把 `C` / `gamma` 填到 `SVM.py` 里重训。注意目标函数用的是测试集（没有独立验证集），最优 `C`/`gamma` 有乐观偏差。

### 4. 从图片批量造特征表

`face/data.py` 目前不能直接使用：先把它里面的 `path = "D:\\desktop\\-Dlib--2.0\\数据集\\S005_001_00000009.png"` 改成真实路径、把 `sheet.write(...)` 取消注释并补上 `book.save("xxx.xls")`，再挂到 `faceEmotion` 实例上调用（它用 `self.detector` / `self.predictor`，需要先 `get_detector()`）。

## 已知问题

> 保留历史问题原文；已修掉的条目在末尾标 **（已修复）** 并说明新做法，未标记的仍然存在。汇总见文末「修复记录」。

- **`data.py` 是跑不通的草稿**：硬编码 Windows 路径 `D:\desktop\-Dlib--2.0\数据集\...`；`book.save()` 缺参数会 `TypeError`；`sheet.write(...)` 全被注释，即使不报错也不会写入任何数据；只处理单张图，没有目录遍历。它更多是"特征定义的可执行说明"。（未修复）
- **`svm_model.pkl` 里不是 SVM**：文件内是 `sklearn.linear_model._logistic`（1.1 KB），因为 `LogisticRegression.py` 也用同一个文件名 `joblib.dump`，把 SVM 的产物覆盖了。要用真正的 SVM 模型，得给两个脚本改成不同文件名后重跑。（未修复）
- **特征列数在脚本之间不一致**：`SVM.py` / `DecisionTreeClassifier.py` 用 6 个特征（含 `brow_k`），`KNN.py` / `LogisticRegression.py` 只用 5 个（`df.columns[0:5]` / `usecols="A:E"`）；而推理端 `face_pd.py` 也是 5 个。想横向比准确率，前提是先把输入维度统一。
  **（`pso.py` 这一处仍在，未修）**：`pso.py` 读的仍是 `df.columns[0:5]`（5 个），而同目录 `SVM.py` 用 `0:6`（6 个，与 `数据集/说明.txt` 的「6 个特征 + 标签」一致）。这处不一致**保留未改**，改 `pso.py` 的列会连带改变它的搜索空间，属于另一次改动。
- **训练/测试数据混用**：`SVM.py` 在 `test_dataset2.xlsx` 上做十折交叉验证并据此保存模型（等于在测试集上训练），而 KNN 用的是 `数据.xlsx`（728 行、只有 5 类）。四个脚本的训练集不同、样本量差 28 倍，脚本里打印的准确率**不能直接横向比较**。（未修复）
- **`数据.xlsx` 缺两个类别**：标签只有 `1, 2, 5, 6, 7`，没有 `3（恶心）`、`4（害怕）`，所以用这份数据训出的 `knn_model.pkl` 永远不会输出这两个类别；而 `face_pd.py` 的 `emo` 表仍是 7 项，`emo[2]`/`emo[3]` 都是 `'nature'` 占位。（未修复）
- **标签语义与 `说明.txt` 不一致**：说明写 `2.蔑视 / 3.恶心 / 4.害怕 / 6.悲伤`，`face_pd.py` 的 `emo` 却是 `sadness / nature / nature / normal`，两套定义没有对齐。（未修复）
- **`pso.py` 结尾会崩**：`print("最佳参数组合：", c, g)` 里的 `g` 未定义，`optimizer.optimize()` 返回的 `(best_cost, best_pos)` 已赋给 `c`，跑到最后一行必 `NameError`；目标函数里 `accuracy` 用的是测试集，等于用测试集当优化目标（没有独立验证集），最优 `C/gamma` 有乐观偏差。**（崩溃已修复，乐观偏差仍在）** 末行改为解包 `(best_cost, best_pos)` 后打印 `最佳参数: C=<C>, gamma=<gamma>, 最优得分=<best_cost>`；"用测试集当优化目标"这一点没动。
- **PSO 目标函数把参数向量当标量用**（这轮额外发现的 bug）：PySwarms 传进来的是 `(n_particles, dimensions)` 矩阵，原代码 `C = params[0]` 实际取到的是「第一个粒子的 `(C, gamma)` 向量」，`SVC(C=array(...))` 会直接抛错。**（已修复）** 目标函数改为 `np.atleast_2d` 归一化后**逐粒子遍历**，每个粒子单独拟合 SVC，返回长度等于粒子数的代价数组（同时兼容 1-D 单向量调用）。
- **模型每帧重复加载**：`face_pd.py` 的 `joblib.load("knn_model.pkl")` 在主循环里每帧每个人脸都执行一次，应从磁盘读一次后缓存。（未修复）
- **`cv2.waitKey` 调了三次**（30 / 100 / 100），按键可能被前一次调用吃掉，`S`/`ESC` 响应不稳定。（未修复）
- **`.pkl` 版本兼容**：模型由老版 scikit-learn 序列化，新版加载会出现 `InconsistentVersionWarning`，极端情况下列顺序/属性不匹配会导致预测异常。（未修复）
- **大文件入库**：`shape_predictor_68_face_landmarks.dat`（约 96 MB）与 6 份 xlsx 全部进 Git，未用 Git LFS。（未修复）
- **`main.py` 里的 `import data` 是死导入，但仍有副作用**：`data` 模块从未被调用，可它顶部 `import xlwt`，若没装 `xlwt`，连实时识别都起不来（`ModuleNotFoundError: xlwt`），装完也会白白拖慢启动。（未修复）
- **无 `requirements.txt`**、`.idea/` 工程文件入库。（未修复）

## 修复记录

| 文件 | 修了什么 | 现在的做法 |
| --- | --- | --- |
| `face/pso.py` | 末行 `print(..., c, g)` 里 `g` 未定义，必 `NameError` | 删除该行，改为解包后打印 `最佳参数: C=..., gamma=..., 最优得分=...` |
| `face/pso.py` | `optimizer.optimize()` 返回 `(best_cost, best_pos)` 却被整体赋给 `c`，拿不到最优参数 | `best_cost, best_pos = result`（兼容只返回 `best_cost` 的旧版本，回退取 `optimizer.swarm.best_pos`），按 `best_pos[0]=C`、`best_pos[1]=gamma` 输出 |
| `face/pso.py` | 目标函数 `C = params[0]` 取到整个粒子向量，`SVC(C=array)` 必崩 | 逐粒子遍历、返回代价数组，兼容单向量调用 |
| `face/pso.py` | `from sklearn.datasets import load_iris` 未使用 | 删除 |
| `face/pso.py` | 特征列 `0:5` 与 `SVM.py` 的 `0:6` 不一致 | **保留未改**，代码里加了注释说明属另一处改动 |

