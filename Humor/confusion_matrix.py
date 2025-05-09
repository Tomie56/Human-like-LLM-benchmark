import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os

def load_result_file(file_path):
    """Load result file and return DataFrame"""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def calculate_metrics(df, model_name):
    """Calculate model evaluation metrics"""
    # Exclude possible error values
    valid_df = df[df["Label_2"] != "error"].copy()
    
    # Get true labels and predicted labels
    y_true = valid_df["Label"].values
    y_pred = valid_df["Label_2"].values
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=["good", "bad"])
    
    # Calculate other metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="good")
    recall = recall_score(y_true, y_pred, pos_label="good")
    f1 = f1_score(y_true, y_pred, pos_label="good")
    
    # Calculate separate accuracy for good and bad labels
    good_df = valid_df[valid_df["Label"] == "good"]
    bad_df = valid_df[valid_df["Label"] == "bad"]
    
    good_accuracy = (good_df["Label"] == good_df["Label_2"]).mean()
    bad_accuracy = (bad_df["Label"] == bad_df["Label_2"]).mean()
    
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "good_accuracy": good_accuracy,
        "bad_accuracy": bad_accuracy,
        "confusion_matrix": conf_matrix
    }
    
    return metrics

def plot_confusion_matrix(conf_matrix, model_name, ax=None):
    """Plot confusion matrix for a single model"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["good", "bad"], yticklabels=["good", "bad"], ax=ax)
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix: {model_name}")
    
    return ax

def compare_models(metrics_list):
    """Compare performance of multiple models"""
    # Create performance comparison table
    models_df = pd.DataFrame([
        {
            "Model": m["model_name"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 Score": m["f1"],
            "Good Class Acc": m["good_accuracy"],
            "Bad Class Acc": m["bad_accuracy"]
        } 
        for m in metrics_list
    ])
    
    # Print comparison table
    print("Model Performance Comparison:")
    print(models_df.to_string(index=False))
    
    # Create bar chart comparison
    plt.figure(figsize=(14, 8))
    
    # Set bar positions
    x = np.arange(len(models_df["Model"]))
    width = 0.15
    
    # Plot bars for each metric
    plt.bar(x - 2*width, models_df["Accuracy"], width, label="Accuracy")
    plt.bar(x - width, models_df["Precision"], width, label="Precision")
    plt.bar(x, models_df["Recall"], width, label="Recall")
    plt.bar(x + width, models_df["F1 Score"], width, label="F1 Score")
    plt.bar(x + 2*width, models_df["Good Class Acc"], width, label="Good Class Acc")
    
    # Add labels and legend
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Performance Metrics Comparison Across Models")
    plt.xticks(x, models_df["Model"])
    plt.legend()
    plt.ylim(0, 1)
    
    plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot confusion matrices for all models
    n_models = len(metrics_list)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for i, metrics in enumerate(metrics_list):
        plot_confusion_matrix(metrics["confusion_matrix"], metrics["model_name"], ax=axes[i])
    
    # Hide extra subplots if number of subplots is greater than number of models
    for i in range(n_models, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return models_df

def main():
    """Main function: Load all result files and compare model performance"""
    # Define mapping between result files and model names
    result_files = {
        "Hunyuan": "chumor.2.0_Hunyuan.tsv",
        "Qwen2.5": "chumor.2.0_with_label2_qwen.tsv",  
        "ERNIE4.0": "chumor.2.0_ernie.tsv",
        "Llama3.1": "chumor.2.0_llama.tsv",
        "GLM4Plus": "chumor.2.0_GLM4Plus.tsv"
    }
    
    metrics_list = []
    
    # Load each file and calculate metrics
    for model_name, file_name in result_files.items():
        file_path = os.path.join(os.getcwd(), file_name)
        
        if os.path.exists(file_path):
            df = load_result_file(file_path)
            if df is not None and "Label" in df.columns and "Label_2" in df.columns:
                metrics = calculate_metrics(df, model_name)
                metrics_list.append(metrics)
                print(f"Loaded and calculated metrics for {model_name}")
            else:
                print(f"File {file_name} has incorrect format or missing label columns")
        else:
            print(f"File {file_path} does not exist")
    
    # Compare models
    if metrics_list:
        comparison_df = compare_models(metrics_list)
        # Save comparison results to CSV
        comparison_df.to_csv("model_comparison.csv", index=False)
        print("Model comparison results saved to model_comparison.csv")
    else:
        print("No valid model results to compare")

if __name__ == "__main__":
    main()