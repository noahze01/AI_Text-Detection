import os
import torch
import argparse
import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import pipeline,AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
# from utils.model_utils import adjust_prediction_score

# DesklibAIDetectionModel类
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config)
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the transformer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

# 上面那个类的训练方法
def predict_single_text(text, model, tokenizer, device, max_len=768, threshold=0.5):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()

    # 返回人的概率
    return 1 - probability



def get_opts():
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--your-team-name",
        type=str,
        default="galaxy",
        help="Your team name"
    )
    # 这里修改测试集位置
    arg.add_argument(
        "--data-path",
        type=str,
        help="Path to the CSV dataset",
        default="/root/autodl-tmp/AI_Text_Detection/UCAS_AISAD_TEXT-test1.csv"
    )
    # 三个模型的集成学习
    arg.add_argument(
        "--model-type",
        type=str,
        choices=[ "your model"],
        help="Type of model to use",
        default="Ensemble Learning"
    )
    # 保存到的目录
    arg.add_argument(
        "--result-path",
        type=str,
        help="Path to save the results",
        default="./result"
    )
    opts = arg.parse_args()
    return opts

def get_dataset(opts):
    print(f"Loading dataset from {opts.data_path}...")
    data = pd.read_csv(opts.data_path)

    # New format: prompt, text
    dataset = data[['prompt', 'text']].dropna().copy()
    print(f"Prepared dataset with {len(dataset)} prompts")
    
    return dataset

def get_model(opts):
    print(f"Loading {opts.model_type} detector model...")
    
    '''
    You should load your model here!
    '''
    
    model = []
    model.append(pipeline(
        "text-classification",
        model="Hello-SimpleAI/chatgpt-detector-roberta",
        device=0,
        truncation=True,
        max_length=512,
        return_all_scores=False))
    
    model.append(pipeline(
        "text-classification", 
        model="raj-tomar001/LLM-DetectAIve_deberta-base",
        device=0,
        truncation=True,
        max_length=512,
        return_all_scores=False))
    

    
    print("Model loaded successfully")
    return model

def run_prediction(model, dataset, model_type):
    print("Starting prediction process...")
    prompts = dataset['prompt'].tolist()
    texts = dataset['text'].tolist()
    
    text_predictions = []
    
    start_time = pd.Timestamp.now()
    
    chatgpt_detector_roberta = model[0]
    LLM_DetectAlve_deberta_base = model[1]
    
    
    # --- Load tokenizer and model ---
    # --- Model and Tokenizer Directory ---
    model_directory = "desklib/ai-text-detector-v1.01"
    ai_text_detector_tokenizer = AutoTokenizer.from_pretrained(model_directory)
    ai_text_detector_model = DesklibAIDetectionModel.from_pretrained(model_directory)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_text_detector_model.to(device)

    
    # Process texts
    print("Processing texts...")
    for text in tqdm(texts, desc="Text predictions"):
        try:
            
            # chatgpt_detector_roberta
            prediction = chatgpt_detector_roberta(text)[0]
            
            if(prediction['label'] == "Human"):
                chatgpt_detector_roberta_final_score = prediction['score']
            else:
                chatgpt_detector_roberta_final_score = 1 - prediction['score']
            
            # LLM_DetectAlve_deberta_base
            prediction = LLM_DetectAlve_deberta_base(text)[0]
            
            if(prediction['label'] == "LABEL_0" or prediction['label'] == "LABEL_1"):
                LLM_DetectAlve_deberta_base_final_score = prediction['score']
            else:
                LLM_DetectAlve_deberta_base_final_score = 1 - prediction['score'] 
            
            
            # ai_text_detector_model
            ai_text_detector_final_score = predict_single_text(text, ai_text_detector_model, ai_text_detector_tokenizer, device)
            
            
            final_score = 0.3332*chatgpt_detector_roberta_final_score+0.3297*LLM_DetectAlve_deberta_base_final_score+0.3371*ai_text_detector_final_score
                
            text_predictions.append(final_score)
            
            
        except Exception as e:
            print(f"Error processing text: {str(e)[:100]}...")
            text_predictions.append(None)
    
    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Create results in the requested format
    results_data = {
        'prompt': prompts,
        'text_prediction': text_predictions
    }
    
    # Create results dictionary
    results = {
        "predictions_data": results_data,
        "time": processing_time
    }
    
    print(f"Predictions completed in {processing_time:.2f} seconds")
    return results

if __name__ == "__main__":
    opts = get_opts()
    dataset = get_dataset(opts)
    model = get_model(opts)
    results = run_prediction(model, dataset, opts.model_type)
    
    # Save results
    os.makedirs(opts.result_path, exist_ok=True)
    writer = pd.ExcelWriter(os.path.join(opts.result_path, opts.your_team_name + ".xlsx"), engine='openpyxl')
    
    # Create prediction dataframe with the required columns
    prediction_frame = pd.DataFrame(
        data = results["predictions_data"]
    )
    
    # Filter out rows with None values
    prediction_frame = prediction_frame.dropna()
    
    time_frame = pd.DataFrame(
        data = {
            "Data Volume": [len(prediction_frame)],
            "Time": [results["time"]],
        }
    )
    
    prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
    time_frame.to_excel(writer, sheet_name="time", index=False)
    writer.close()
    
    print(f"Results saved to {os.path.join(opts.result_path, opts.your_team_name + '.xlsx')}")
