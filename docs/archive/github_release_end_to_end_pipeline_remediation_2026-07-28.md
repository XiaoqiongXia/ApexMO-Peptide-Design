# AMP Design GitHub 端到端发布修复报告

- 日期：2026-07-28
- 对应审查：`docs/github_release_end_to_end_pipeline_code_review_2026-07-28.md`
- 修复范围：采样、候选准备、评分、Pareto 初始化、遗传搜索、最终合并、硬门槛导出、配置、发布构建和 CI。

## 已完成的核心修复

1. 新增 `amp-design prepare-candidates`：逐 shard 验证、SQLite 精确去重、流式写出 Parquet，并记录输入 shard 哈希和淘汰计数。
2. 新增 `amp-design score-candidates`：统一执行 ESM-2 toxicity/hemolysis、便携 ESM 5-NN 安全适用域、APEX 40-model ensemble、不确定性、MMseqs2 新颖性和探索性稳定性评分。
3. 新增 `amp-design initialize-pareto`：只从本次新采样评分表构建 Pareto ranked pool 和新的 cluster-diverse 初始种群。
4. 遗传 offspring 新增外部禁止集合：训练集、完整原始 candidate pool 和配置的额外 novelty reference tables 均不能作为 exact duplicate offspring。
5. 遗传可行性统一加入 safety AD、APEX uncertainty 和 MMseqs2 novelty；初始种群和新 offspring 使用同一套正式约束。
6. 新增 `amp-design finalize`：合并各 seed 完整 evaluation cache，重新计算联合 Pareto 前沿并应用最终 cluster cap。
7. 新增 `amp-design hard-filter`：统一输出 audit CSV、最终 CSV、FASTA 和 manifest；硬门槛包括 MIC、toxicity、hemolysis、safety AD、APEX uncertainty、novelty 和可选 stability。
8. 新增 `amp-design pipeline`：从 sampling 到最终 export 的单命令、分阶段可恢复流程。
9. 移除采样器原 100-shard 上限；前 100 个 shard 的历史 generation seed 保持不变，更高 shard id 使用稳定 SHA-256 派生 seed。
10. 修复 `init-config` 模板，使其指向发布仓库实际模型和资产，并增加完整 pipeline 模板。
11. 发布环境改用 PyPI 的 Flow Matching/pymoo，加上 optimization extras 和 MMseqs2，不再引用 setup 脚本不会创建的 editable external 目录。
12. 发布 CI 安装 optimization extras；手工 release-assets job 拉取 Git LFS、验证资产并执行 sampling preflight。
13. GitHub release builder 新增 safety AD reference assets，并停止打包不可运行的旧版离线/收尾脚本。

## 新颖性定义

默认 `novelty_references` 为空时，新颖性参照冻结 AMP 训练集。用户可以在 optimization 配置中加入一个或多个带 `sequence` 列的 CSV/Parquet 表，例如获准使用的已知 AMP 数据库导出。

最终的 `novelty_hard_threshold_pass` 表示在配置的 reference universe 中没有达到 `novelty_max_identity` 与 `novelty_min_coverage` 的 MMseqs2 hit。软件不会在未提供外部数据库时声称对所有公开数据库全局新颖。

## 稳定性定位

稳定性 Ridge 模型继续标记为 exploratory。它预测 `log10(T1/2 hours)`，默认最终门槛为预测半衰期至少 1 小时。稳定性不进入三目标 Pareto 排序，避免把未正式 promotion 的模型提升为搜索目标。

## 真实缩小版闭环验证

使用正式 seed-42 Flow Matching checkpoint、正式 ESM-2、正式 toxicity/hemolysis 模型、正式 safety AD reference、正式 stability model 和全部 40 个 APEX 模型执行：

- 采样并准备：220 条；
- 完整评分：220 条；
- 正式 Pareto eligible：97 条；
- 新初始种群：5 条；
- 遗传搜索：1 seed × 1 generation，population 5；
- 联合 rank-0：26 条；
- 最终 80% identity cluster cap 后：25 条；
- 同时通过全部七项硬门槛：0 条。

0 条是合法且重要的结果。小样本中各单项门槛均有通过者，但没有序列同时满足所有门槛；实现没有自动放宽阈值。完整正式运行应扩大采样池，同时保留相同 frozen thresholds。

## 尚需项目所有者决定

项目级 `LICENSE` 仍不能由代码修复自动决定。Flow Matching 为 CC BY-NC，APEX 为非营利研究用途；项目所有者需要选择相容的非商业研究许可证或取得第三方额外授权。许可证确定前，仓库可以公开为 source-available research code，但不能宣称为标准开源软件。
