import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def eval_func(gt_name="ground_truth", path=""):
    """
    Args:
        gt_name (str): Name of the ground truth file (without extension)
        path (str): Path to directory containing result files
        
    Returns:
        dict: Dictionary with team/method names as keys and their metrics as values
    """
    ret = {}

    # Get all files in the directory
    teams = os.listdir(path)
    
    # Filter out ground truth and leaderboard files
    if gt_name + ".csv" in teams:
        teams.remove(gt_name + ".csv")
    if "LeaderBoard.xlsx" in teams:
        teams.remove("LeaderBoard.xlsx")

    # Read ground truth labels
    gts = pd.read_csv(
        os.path.join(path, gt_name + ".csv")
    )
    
    print(f"Found {len(teams)} team/method submissions to evaluate")
    
    for team in teams:
        try:
            # Read prediction results
            data = pd.read_excel(
                os.path.join(path, team),
                sheet_name="predictions",
                engine="openpyxl"
            )
            
            # Extract prediction values
            predictions = data["text_prediction"].values
            
            # Get ground truth labels
            ground_truth = gts["label"].values
            
            # Convert prediction probabilities to binary classification results (threshold 0.5)
            binary_pred = (predictions >= 0.5).astype(int)
            
            # Calculate AUC
            auc = roc_auc_score(ground_truth, predictions)
            
            # Calculate F1 score
            f1 = f1_score(ground_truth, binary_pred)
            
            # Calculate precision and recall
            precision = precision_score(ground_truth, binary_pred)
            recall = recall_score(ground_truth, binary_pred)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(ground_truth, binary_pred).ravel()
            
            # Calculate accuracy
            accuracy = accuracy_score(ground_truth, binary_pred)  # Convert to percentage
            
            # Read time information
            time_data = pd.read_excel(
                os.path.join(path, team),
                sheet_name="time",
                engine="openpyxl"
            )
            
            # Calculate average processing time per sample
            mean_time = time_data["Time"][0] / time_data["Data Volume"][0]
            
            # Store results
            ret[team.split(".")[0]] = {
                "auc": auc,
                "accuracy": accuracy,  # Acc.(%)
                "f1": f1,             # F1
                "fn": fn,             # FN
                "fp": fp,             # FP
                "precision": precision, # Prec
                "recall": recall,     # Rec
                "mean_time": mean_time
            }
            
            print(f"Evaluated {team}: AUC={auc:.4f}, F1={f1:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")
            
        except Exception as e:
            print(f"Error processing {team}: {str(e)}")
            continue

    return ret

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--submit-path",
        type=str,
        default="/root/autodl-tmp/AI_Text_Detection/result",
        help="Path to directory containing submission files"
    )
    arg.add_argument(
        "--gt-name",
        type=str,
        default="/root/autodl-tmp/AI_Text_Detection/UCAS_AISAD_TEXT-val",
        help="Name of ground truth file (without extension)"
    )
    opts = arg.parse_args()
    
    results = eval_func(gt_name=opts.gt_name, path=opts.submit_path)

    # Create leaderboard Excel file
    writer = pd.ExcelWriter(os.path.join(opts.submit_path, "LeaderBoard.xlsx"), engine="openpyxl")
    
    leaderboard_data = {
        "Team/Method": results.keys(),
        "AUC": [res["auc"] for res in results.values()],
        "Acc": [res["accuracy"] for res in results.values()],
        "F1": [res["f1"] for res in results.values()],
        # "FN": [res["fn"] for res in results.values()],
        # "FP": [res["fp"] for res in results.values()],
        # "Prec": [res["precision"] for res in results.values()],
        # "Rec": [res["recall"] for res in results.values()],
        "Avg Time (s)": [res["mean_time"] for res in results.values()]
    }
    
    # Create DataFrame and calculate weighted score
    leaderboard_df = pd.DataFrame(data=leaderboard_data)
     
    # Calculate weighted score (AUC weight 0.6, F1 weight 0.25, accuracy weight 0.1)
    leaderboard_df["Weighted Score"] = (
        0.6 * leaderboard_df["AUC"] + 
        0.3 * leaderboard_df["Acc"] + 
        0.1 * leaderboard_df["F1"] / 100 
    )
    
    # Sort by weighted score in descending order
    leaderboard_df = leaderboard_df.sort_values(by="Weighted Score", ascending=False)
    
    leaderboard_df.to_excel(writer, index=False)
    writer.close()
    
    print(f"Extended leaderboard saved to {os.path.join(opts.submit_path, 'LeaderBoard.xlsx')}")
