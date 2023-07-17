# HW-ICT-ProcedureKE
使用传统序列标注模型进行流程知识抽取。方法分为两阶段，第一阶段进行动作识别，第二阶段进行语义角色识别。使用BERT+MLP单层分类器进行动作识别，再使用BERT或BERT_GRU进行语义角色识别。
## 使用说明
运行code/PR.py进行动作识别，运行code/BERT_GRU_multi.py进行语义角色识别
## 功能更新
增加了模型评测功能