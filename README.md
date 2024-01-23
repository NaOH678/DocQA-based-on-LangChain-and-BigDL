# DocQA-based-on-LangChain-and-BigDL
这个仓库存放了基于大语言模型chatGLM-6B搭建的文档问答应用。通过LangChain开发，BigDL加速。

## 环境搭建
CentOS 8.4 64位
CPU Intel/CascadeLake 32核 64G
系统盘30G
数据盘60G

请根据报告中的方法安装所需的依赖

或者使用environments.yml安装环境
```
conda env create -f environment.yml
conda activate llm-tutorial
```

执行代码

```
python gradio_en/test_gradio.py
```
另外需要注意模型要自己在文件夹中放好。模型太大无法上传。
文件路径按照自己的文件进行设置即可。

效果展示
<img width="1349" alt="image" src="https://github.com/NaOH678/DocQA-based-on-LangChain-and-BigDL/assets/112929756/e4e4da0c-abdb-43ab-9767-629cfd74c333">
