# 第三方数据声明

本仓库的代码以 [MIT](../LICENSE) 许可发布。
**但项目使用的数据集是第三方的，许可是另外一回事**——这里把它讲清楚。

---

## IBM AMLSim Example Dataset

| 项 | 内容 |
|---|---|
| 来源 | Kaggle `anshankul/ibm-amlsim-example-dataset` |
| 许可 | **Apache License 2.0** |
| 是否包含在本仓库 | **否**——由 `scripts/fetch_data.py` 按需下载 |

数据集由 [IBM AMLSim](https://github.com/IBM/AMLSim) 项目生成。
该项目的定位是：提供**多智能体银行交易模拟器**，
生成合成银行交易数据以及一组已知的洗钱模式，
**主要用于测试机器学习模型和图算法**。

### 为什么数据不入库

1. **体积**：压缩 11.9 MB / 解压 61.5 MB；
2. **可重建**：一条命令就能拿到，且 `fetch_data.py` 里写死了期望行数——
   数据换了会立刻发现；
3. **许可边界清晰**：代码是 MIT，数据是 Apache 2.0，**不混在一起**。

---

## 关于许可识别

GitHub 的许可识别器要求 `LICENSE` 文件是**干净的标准文本**。
本仓库因此把第三方声明放在这个独立文件里，
而不是追加在 `LICENSE` 末尾——追加会让 GitHub 显示 `NOASSERTION` 而不是 `MIT`。
