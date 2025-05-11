# AI_Text Detection

国科大2025年春季学期人工智能对抗比赛文本检测赛道的作品

## 运行项目

```bash
python prediction.py \
    --data-path $YOUR_DATASET_PATH/test1 \
    --result-path $YOUR_SAVE_PATH/
```

特别说明：

1. $YOUR_DATASET_PATH/test1为UCAS_AISAD_TEXT-test1.csv的路径，不要加后缀csv
2. $YOUR_SAVE_PATH，结果为execl文件，文件格式如下：

Excel 文件（galaxy.xlsx,我们队伍的名称），包含两个工作表：

1. **predictions 工作表**

   - 包含以下内容：

     - prompt：原始提示文本
     - text_prediction：人类作者身份的概率（数值越高，越可能是人类作者）

   - 示例：

     ```bash
     prompt,text_prediction
     "解释量子计算",0.95
     "描述气候变化",0.68
     ...
     ```

2. **time 工作表**

   - 包含以下内容：

     - Data Volume：处理的样本数量
     - Time：总处理时间（单位：秒）

   - 示例：

     ```bash
     Data Volume,Time
     "6000",53.21
     ```

