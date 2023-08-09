import pandas as pd
import requests

def get_need_and_not_list():
  sheet_id = "18OtMuMx1-UUHEhz9bLVizqOwW4k04opMQo9kTKWcrcQ"
  sheet_name = "skill_remove_and_add"
  url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
  data = pd.read_csv(url)
  data['想增加的技能（技能與技能間請以","隔開）'] = data['想增加的技能（技能與技能間請以","隔開）'].astype(str)
  data['想要移除的預測結果（請以","隔開）'] = data['想要移除的預測結果（請以","隔開）'].astype(str)
  need_list = [item.strip() for need in data['想增加的技能（技能與技能間請以","隔開）'] for item in need.split(",")]
  delete_list = [item.strip() for delete in data['想要移除的預測結果（請以","隔開）'] for item in delete.split(",")]
  return need_list, delete_list

def add_data_to_sheet(sentences, predict_entities):
  url = 'https://script.google.com/macros/s/AKfycbxZrEFWLRBz7mA--i8hweW4UWlInDPNOseD7zVtzNSaHXxSPxMDmBwpdsF7bRYKAbCE/exec'

  # JSON data payload
  data = {
    "job_description": sentences,
    "result": str(predict_entities)
  }

  # Make the POST request
  response = requests.post(url, json=data)

  # Print the response
  print(response.text)