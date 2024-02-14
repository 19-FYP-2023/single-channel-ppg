import os
import pandas as pd

data_entc = pd.read_csv("dataset/paraqum-dataset/good_signal_labels.csv")
paths = ["dataset/paraqum-dataset/data-captures/data02052019/sensor2","dataset/paraqum-dataset/data-captures/data02052019", "dataset/paraqum-dataset/data-captures/data09012019", "dataset/paraqum-dataset/data-captures/data16012019" ]

data_entc["Path"] = ""
for index, row in data_entc.iterrows():

    for path in paths:
        count = 0
        if os.path.exists(f"{path}/{row['File Name']}"):
            data_entc.at[index, "Path"] = f"{path}/{row['File Name']}"

            
data_entc = data_entc[data_entc["Tag"] == "GOOD"]
data_entc.to_csv("dataset/paraqum-dataset/good_signal_labels_with_paths.csv", index=False)