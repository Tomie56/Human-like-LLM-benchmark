


import os
import pandas as pd
from tqdm import tqdm
import time
#from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser



# 配置API密钥 (可以放在.env文件中)
os.environ["OPENAI_API_KEY"] = ""
# GLM API 端点
os.environ["OPENAI_API_BASE"] =  "https://api.hunyuan.cloud.tencent.com/v1"

def setup_joke_evaluator():
    """
    设置用于评估笑话解释的Langchain组件
    """
    # 创建提示模板
    template = """
你将看到一个笑话以及对这个笑话的解释。
请判断这个解释是否完全解释了笑话。根
据判断，选择"完全解释"或"部分/没有解
释"，不需要解释为什么对或者不对。完全解释输出"good"，部分/没有解释输出"bad"。
笑话：{joke}
笑话解释：{explanation}
"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["joke", "explanation"]
    )
    
    # 设置SiliconFlow DeepSeek-V3模型
    llm = ChatOpenAI(
        model="hunyuan-turbos-latest",
        temperature=0.0,  # 使用确定性输出
        max_tokens=10     # 我们只需要简短回复
    )
    
    # 解析模型输出
    output_parser = StrOutputParser()
    
    # 创建简单的Langchain链
    chain = prompt | llm | output_parser
    
    return chain

def process_tsv_file(input_file="chumor.2.0.tsv", output_file="chumor.2.0_with_label2.tsv"):
    """
    处理TSV文件，评估每个笑话解释，并添加Label_2列
    """
    print(f"开始处理文件: {input_file}")
    
    # 读取TSV文件
    df = pd.read_csv(input_file, sep='\t')
    
    # 设置评估链
    joke_evaluator = setup_joke_evaluator()
    
    # 添加新列用于评估结果
    df["Label_2"] = None
    
    # 处理每一行
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="评估笑话解释"):
        joke = row["Joke"]
        explanation = row["Explanation"]
        
        try:
            # 调用评估链
            result = joke_evaluator.invoke({
                "joke": joke,
                "explanation": explanation
            })
            
            # 处理结果 - 仅保留"good"或"bad"
            if "good" in result.lower():
                
                label = "good"
                #print("good")
            elif "bad" in result.lower():
                #print("bad")
                label = "bad"
            else:
                print(f"行 {idx+1}: 意外响应 '{result}'，默认设为'bad'")
                label = "bad"
                
            # 保存结果
            df.at[idx, "Label_2"] = label
            
            # 每10条保存一次进度
            # if (idx + 1) % 10 == 0:
            #     df.to_csv(output_file, sep='\t', index=False)
            #     print(f"已保存进度至第 {idx+1} 行")
                
            # 小延迟以避免API速率限制
            time.sleep(0.5)
            
        except Exception as e:
            print(f"处理行 {idx+1} 时出错: {e}")
            df.at[idx, "Label_2"] = "error"
            
            # 出错时也保存进度
            df.to_csv(output_file, sep='\t', index=False)
    
    # 保存最终结果
    df.to_csv(output_file, sep='\t', index=False)
    print(f"处理完成! 结果已保存到 {output_file}")
    
    return df

def calculate_accuracy(df):
    """
    计算模型评估准确率
    
    参数:
        df (DataFrame): 包含Label和Label_2的数据框
        
    返回:
        dict: 包含准确率指标的字典
    """
    print("\n==== 准确率评估 ====")
    
    # 排除可能的错误值
    valid_df = df[df["Label_2"] != "error"].copy()
    
    # 计算总体准确率
    correct_predictions = (valid_df["Label"] == valid_df["Label_2"]).sum()
    total_samples = len(valid_df)
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    print(f"总体准确率: {overall_accuracy:.4f} ({correct_predictions}/{total_samples})")
    
    # 计算每个标签的准确率
    label_metrics = {}
    for label in ["good", "bad"]:
        label_df = valid_df[valid_df["Label"] == label]
        label_correct = (label_df["Label"] == label_df["Label_2"]).sum()
        label_total = len(label_df)
        label_accuracy = label_correct / label_total if label_total > 0 else 0
        
        label_metrics[label] = {
            "accuracy": label_accuracy,
            "correct": label_correct,
            "total": label_total
        }
        
        print(f"标签 '{label}' 准确率: {label_accuracy:.4f} ({label_correct}/{label_total})")
    
    # 创建混淆矩阵
    confusion_matrix = pd.crosstab(valid_df["Label"], valid_df["Label_2"], 
                                    rownames=["真实标签"], colnames=["预测标签"])
    print("\n混淆矩阵:")
    print(confusion_matrix)
    
    # 返回指标字典
    metrics = {
        "overall_accuracy": overall_accuracy,
        "label_metrics": label_metrics,
        "confusion_matrix": confusion_matrix
    }
    
    return metrics

if __name__ == "__main__":
    INPUT="chumor.2.0.tsv"
    OUTPUT="chumor.2.0_Hunyuan.tsv"
    results_df=process_tsv_file(input_file=INPUT, output_file=OUTPUT)
    accuracy_metrics = calculate_accuracy(results_df)