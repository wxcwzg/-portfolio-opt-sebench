FROM python:3.11-slim

WORKDIR /home/workspace/portfolio_opt

RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scipy==1.13.0 scikit-learn==1.4.2

COPY agent-start/ /home/workspace/portfolio_opt/
COPY scorer/ /home/workspace/portfolio_opt/scorer/
