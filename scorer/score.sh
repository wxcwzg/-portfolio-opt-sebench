#!/bin/bash
# 投资组合优化评分脚本
# 由Judge容器调用，读取Agent提交的weights.csv，计算夏普比率并输出分数
cd /home/workspace/portfolio_opt
python3 /home/workspace/portfolio_opt/scorer/evaluate.py
