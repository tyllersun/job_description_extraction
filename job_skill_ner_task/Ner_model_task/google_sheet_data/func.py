import pandas as pd

def get_need_and_not_list():
  sheet_id = "18OtMuMx1-UUHEhz9bLVizqOwW4k04opMQo9kTKWcrcQ"
  sheet_name = "skill_remove_and_add"
  url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
  data = pd.read_csv(url)
  need_list = [item.strip() for need in data['想增加的技能（技能與技能間請以","隔開）'] for item in need.split(",")]
  delete_list = [item.strip() for delete in data['想要移除的預測結果（請以","隔開）'] for item in delete.split(",")]
  return need_list, delete_list