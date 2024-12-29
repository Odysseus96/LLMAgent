### LLMAgent（名称暂定） 
计划开发基于LLM的Agent  
1. 由于多模态大模型还难以在特定的视觉场景落地，而特定场景在短时间又缺乏足够的数据训练相关多模态大模型。  
2. 因此本仓库计划复现React范式，结合LLM的planning能力，探索其在特定行觉场景的应用。  
### 开发过程
计划基于langchain实现一个简单的agent，给LLM加入视觉的能力。  

### 计划
- [x] 实现工具api  
- [x] 基于ollama调用本地大模型  
- [x] 实现简单的非视觉类的agent  
- [x] 用langchain封装了ollama，支持stream
- [ ] 实现调用视觉工具的agent
- [ ] 对agent进行封装



**已完成**：  
- 现在已用ReAct构建了完整的Agent，通过调用工具初步实现了谷歌搜索和获取arxiv论文的基本功能。使用方法:
```python
# google search
python test/google_search_agent.py
# arxiv search
python test/arxiv_api_agent.py
```

### 相关论文
- [React](https://arxiv.org/abs/2210.03629)  
- [MM-REACT](https://arxiv.org/abs/2303.11381)