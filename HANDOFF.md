# 水果切割仿真研究交接

更新于 2026-11-26。八种水果模型发布于 v0.1.0。本次 v0.2.0 归档三轮切割物理实验，包括整根黄瓜的切后操纵、CRESSim-MPM 对照，以及均匀果肉试块的力与力矩测量。

## 先找到成果

| 想做什么 | 入口 |
| --- | --- |
| 看半切、全切和切后操纵 | [三轮实验的视频与测量](README.md#cutting-physics-experiments) · [五个可在 Isaac Sim 中打开的回放](https://github.com/zhiyuan-gao/fruit-cutting-research/releases/tag/v0.2.0) |
| 查看当前试块的力和力矩 | [实验图](records/coupon-results.png) · [三组曲线 CSV](records/coupon-wrench) · [参数、验收结果及后续研究](CUTTING_PHYSICS_PLAN.md) |
| 看八种水果的外观与内部结构 | 下载[演示 HTML](demo/fruit-cutting-films.html) 并在浏览器打开。32.25 秒的双倍速合集可离线播放。 |
| 在 Isaac Sim 查看水果模型 | 从 [v0.1.0 Release](https://github.com/zhiyuan-gao/fruit-cutting-research/releases/tag/v0.1.0) 下载模型压缩包。解压后用 **File → Open** 选择任一 USDZ。西瓜在压缩包中叫 `watermelon.usdz`，在生成器中叫 `melon`。 |
| 改代码或复现实验 | 从[运行说明](README.md#reproduce-the-reference-samples)准备 Isaac Sim 5.1.0、CUDA 和 Docker 环境。 |

## 八种水果怎么建模

七种水果沿用 CulinaryCut 的外形与果皮贴图。牛油果复用 RoboNinja 演示里的 EmbodiedSWE 整果 USD，沿用 1.4 倍缩放，补齐缺失的一个三角形。内部组织由本项目在三维体积中分区，再生成 USD 节点、软组织粒子与刚体几何。苹果、橙子和西瓜还参考了 GaussianFluent / FruitNinja 点云；部分果肉颜色细节来自照片及程序化着色。原始外表面本身不能给出整套内部解剖结构。[模型准备代码](scripts/prepare_models.py) · [上游输入及校验值](sources.lock.json)

| 模板 | 当前参考样本的组成与物理处理 |
| --- | --- |
| 香蕉 `banana` | 果皮和果肉是独立节点，梗只属于果皮。参考样本为绿蕉；绿、黄阶段关联外皮颜色和果肉刚度。 |
| 橙子 `orange` | 外皮和十瓣果肉可分别查看，物理分区区分皮、肉和瓣膜。瓣膜有材料标签，参考 USDZ 中共有 11 个网格。当前为无籽橙。 |
| 牛油果 `avocado` | 果皮、果肉与独立椭球硬核。核固定，刀碰核会被挡住。模板有较硬和成熟两个阶段，发布样本为成熟阶段。 |
| 苹果 `apple` | 果皮、果肉、软果芯、独立短柄和五颗可动刚体籽。每颗籽都能按 USD 路径追踪。 |
| 黄瓜 `cucumber` | 果皮、果肉与连续的软籽腔。已按用户要求移除独立籽粒。 |
| 桃子 `peach` | 果皮、果肉、固定桃核与独立短柄。 |
| 草莓 `strawberry` | 外层、红色果肉和中央髓部。表面的瘦果保留在外皮贴图中。 |
| 西瓜 `melon` | 绿色外皮、白色瓜皮层、红瓤和 54 颗独立可动刚体籽。两层瓜皮共享一个显示网格，物理材料分开。英文展示沿用 *watermelon rind*。 |

每种水果的 [模板目录](templates) 保存 `template.json` 和 `anatomy.npz`。前者记录材料范围、单位、依据及成熟度覆盖值，后者保存组织、瓣区标签与体素到 USD 的坐标变换。生成器按种子采样尺寸、轴向比例、锥度、弯曲、内部偏移及材料参数，用同一变形保持内外结构对齐。香蕉和牛油果支持按成熟度采样；其余六种目前只有参考外观。文献实测、跨品种换算和工程估计在模板中分别标明。[生成器](generate.py) · [组织物性测量表](records/fruit-tissue-measurements.csv)

## 切割机制与验证范围

- **上游原版接口。** 早期工作先把 CulinaryCut 的刀具、粒子、状态和接触力接入 Isaac Sim，并与上游按相同子步做对照。香蕉曾做过直切和中途转刀 32° 的试验，退刀 0.5 秒后，在 48 mm 截面测得粒子间隙中心分别为 −0.06 mm 和 +11.32 mm。这证明粒子响应实际刀路，单凭开口画面不能证明永久断裂。这组早期香蕉试验保存在原始工作区的 `isaac-mpm-cutting/docs/handoff.md`。
- **本项目的扩展。** 当前 `FruitSimulation` 默认启用单条切口扩展。算法根据相邻刀具位姿和实际接触的刀刃扫过区域，记录粒子的切口侧别，并在切口邻域分别更新两侧动量。早期压合与回拉试验中，10,357 个标记粒子的侧别保持稳定。已验证范围是一次下切形成的一条弯折切口；处理多条相交切口还需扩展现有算法，按材料设置断裂韧性还需建模和标定。GaussianFluent 提供了三维分区的参考，其碎裂求解器和 β 参数未接入。[切口实现](src/mpm_cutting/fracture.py) · [求解入口](simulate.py)
- **显示与物理。** 分开的 USD 网格提供可见的组件。软组织的密度和弹性按粒子材料编号赋值，硬核和籽另有碰撞几何。Release 的 USDZ 适合直接查看；运行切割还需要生成的 `mpm.npz` 和 `sample.json`。`simulate.py --stage` 导出按组织着色的粒子、刀和刚体时间线。HTML 展示的是另行制作的多镜头成片。
- **运动与参数。** 当前演示由程序规定直刀轨迹，场景有重力和砧板接触。4 m/s 是快速数值测试速度。机器人末端闭环控制、实物刀力标定、籽与果肉黏附，以及橙瓣沿膜自然脱开均是后续课题。用户要求刀柄始终在果体外；切块掉落或翻倒不作为验收条件。

| 已有证据 | 结果 |
| --- | --- |
| [八种参考样本](records/reference-samples.json) · [Isaac Sim 打开检查](records/isaac-stage-checks.json) | 从干净副本重建后，软粒子和刚体数组、外皮贴图哈希与演示样本一致。八个 USDZ 在 Isaac Sim 中均能解析网格和贴图。显示网格与早期样本的最大表面位置差约 1.1 µm，部分果肉顶点颜色略有差异。 |
| [牛油果碰核](records/avocado-through-to-pit.json) | 居中直切时，刃口前进约 22.7 mm 后遇到固定硬核，记录间隙约 0.030 mm。 |
| [苹果籽追踪](records/apple-seed-track.json) | 视频追踪的是 `Seed_004`，首次刀籽接触在 11.34 ms，记录位移 10.83 mm。 |
| [橙子参考刀路](records/orange-through-cut.json) · [刀柄间距](records/gallery-handle-clearance.json) | 无籽橙的归档试验从表皮切至砧板。干净副本另一次复跑导出了 121 帧时间线。八种水果的视频逐帧检查中，刀柄与水果表面的最小水平间距约 46.8 mm。 |

## 黄瓜单刀切块实验

在 Isaac Sim 打开 [黄瓜时间线](https://github.com/zhiyuan-gao/fruit-cutting-research/releases/download/v0.2.0/cucumber.usdc)。归档复跑的第 8 帧为半切，场景显示 `/World/Fruit/Whole`；第 19 帧首次显示 `Piece_001` 和 `Piece_002`。每块下有各组织的粒子节点。每帧对应的物理时间和实验阶段记录在 `/World` 的 `mpm:simulationTime` 和 `mpm:phase`。

固定参考样本、24 mm 宽支撑槽、4 m/s 单次下切，使用刀具网格下缘和注册到世界坐标的砧板 SDF。部分切割仍保留材料邻接路径。切穿后以材料图的两个连通分量给粒子分配块 ID。撤刀后仅对一块施加按质量分配的外力，使用同一撤刀状态的无载分支作对照。复跑的受力块和另一块沿施力方向分别多移动 14.80 mm 和 0.58 mm。第 70 帧检测到切面再接触，第 87 帧再次分开，两块粒子归属稳定。此结果覆盖当前离散分辨率的单刀数值试验。鼠标拖拽、夹爪接触、多刀分块和实物受力验证仍待开发。[逐帧仿真记录](records/cucumber-piece-trial.json) · [实验入口](piece_trial.py) · [运行说明](README.md#single-cut-cucumber-trial-in-isaac-sim)

Isaac Sim 5.1 已打开归档复跑的 88 帧时间线，逐帧检查可见节点，共 114,517 个粒子 ID，分块前后均无遗漏或重复。[切口与粒子的逐帧对照](records/cut-alignment.png) 能看到切面附近有一条细长的粒子带。按初始切面两侧各 25 mm 的范围检查，第 43 帧归入同一果块的 59,830 对原始邻点中，有 1,208 对间距超过 5 mm，最大为 19.33 mm；参考邻接半径为 2.5 mm。后续研究切面完整性时应单独检查这类局部拉伸。

## CRESSim-MPM 黄瓜对照

用原版 GPU MLS-MPM 和同一份 114,517 粒子黄瓜复跑单刀。原生平面切刀的接触厚度为 4 mm，使用 4 m/s 下切。半切撤刀后，两处上层组织各向外移动 4 mm，原始跨切口邻点的中位轴向间距比固定夹持多增加 0.24 mm，底部几乎无变化。全切撤刀后仅给右块 0.6 m/s 的向外速度，扣除无载对照，原生单网格中右、左块分别多移动 8.93 和 8.59 mm。较小时间步下分别为 7.84 和 7.99 mm。切透后采用两个独立网格时，右块多移动 5.72 mm。左块共用同一无载轨迹，额外位移按构造为零；此分支仅检验向外移动。回弹、刀口附近的局部压缩和重复运行稳定性仍限制物理结论。[演示视频](https://github.com/zhiyuan-gao/fruit-cutting-research/releases/download/v0.2.0/cressim-cucumber-results-zh.mp4) · [测量及局限](records/cressim-cucumber.json) · [复现步骤](scripts/cressim_trial/README.md)

## 果肉试块与下一轮工作

固定刀速和材料后，刀在两排粒子中间下切与横移 0.5 mm 下切的做功分别为 2.81 和 8.40 mJ。下一轮要先降低这种对粒子排列的敏感性，再评估实机切割阻力。现有试块使用纯弹性材料和一条竖直刀路。整瓜多组织的新后端、应力驱动的自然裂纹和实机校准仍待验证。[试块记录](records/cutting-physics-coupon.json) · [研究计划](CUTTING_PHYSICS_PLAN.md)

## 从哪里继续

只看成果可直接打开上面的 HTML 或 USDZ。要改模型或运行求解器，先按 [README 的复现步骤](README.md) 下载并校验 62 个固定版本的上游文件、准备 PyBullet、构建镜像、生成基础模型和八份参考样本。以下命令在仓库根目录、完成上述准备后运行。

```bash
# 检查参考样本的网格、粒子、材料与贴图
./python.sh validate.py --reference

# 重跑无籽橙的直切，保存数值与可在 Isaac Sim 中查看的时间线
./python.sh simulate.py examples/orange/2026_0000 --speed 4 --report outputs/orange.json --stage outputs/orange.usdc
./python.sh validate_isaac.py --cut outputs/orange.usdc
```

如果下一步是研究**材料**，先从测量表选定水果、组织和对比指标，保留固定随机种子的参考样本。如果要研究**机器人控刀**，接入点是 `FruitSimulation.command(...)`、`step(ticks)` 和 `state()`，外部坐标为 USD Z-up、长度为米、四元数为 wxyz。当前优先验证试块切割功对数值采样的敏感性。

延续用户已经确认的约定：平滑表面时保留原来的水果轮廓，苹果和桃子的短柄单独建模；展示时区分可见结构与实际接触的结构，八种水果的合集使用英文，切割实验另外提供中文结果视频。Git 保留研究代码、模板、物性记录和最终 HTML；成品 USDZ 随 Release 分发，上游资产按清单获取，历次渲染、剪辑和 HTML 生成脚本留在旧工作区。
