import os
import random
import shutil
from itertools import islice

random.seed(42)
outputFolderPath = "./data/SplitData"
inputFolderPath = "./data/All"
Ratio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]
try:
    shutil.rmtree(outputFolderPath)  # xoa neu no ton tai
except OSError as e:
    os.mkdir(outputFolderPath)  # tao lai folder

# ---directories to create-----
os.makedirs(os.path.join(f"{outputFolderPath}", "train/images"), exist_ok=True)
os.makedirs(os.path.join(f"{outputFolderPath}", "train/labels"), exist_ok=True)
os.makedirs(os.path.join(f"{outputFolderPath}", "val/images"), exist_ok=True)
os.makedirs(os.path.join(f"{outputFolderPath}", "val/labels"), exist_ok=True)
os.makedirs(os.path.join(f"{outputFolderPath}", "test/images"), exist_ok=True)
os.makedirs(os.path.join(f"{outputFolderPath}", "test/labels"), exist_ok=True)
# ---get all image and label paths---
listName = os.listdir(inputFolderPath)
# lay ten file khong co duoi, uniqueNames = set()-> lay ten file khong trung
uniqueNames = []
for name in listName:
    uniqueNames.append(name.split(".")[0])
uniqueNames = list(set(uniqueNames))
print(len(uniqueNames))
# ---shuffle the unique names---
random.shuffle(uniqueNames)
lenData = len(uniqueNames)
lenTrain = int(lenData * Ratio["train"])
lenVal = int(lenData * Ratio["val"])
lenTest = int(lenData * Ratio["test"])
print(f"train size: {lenTrain}, val size: {lenVal}, test size: {lenTest}")
# ---split the unique names into train, val, and test sets---
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)  # biến list thành interator duyệt tuần tự
Output = [
    list(islice(Input, elem)) for elem in lengthToSplit
]  # islice(Input, elem) = lấy elem phần tử tiếp theo
"""
a,b,c,d,e,f
2,2,2
islice(a,2) -> a,b
islice(a,2) -> c,d
"""
print(
    f"Total Images:{lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}"
)
# ---copy the images and labels to their respective directories---
sequence = ["train", "val", "test"]
for i, out in enumerate(Output):
    for fileName in out:
        shutil.copy(
            f"{inputFolderPath}/{fileName}.jpg",
            f"{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg",
        )
        shutil.copy(
            f"{inputFolderPath}/{fileName}.txt",
            f"{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt",
        )

print("Split Process Completed...")
# -------- Creating Data.yaml file  -----------

dataYaml = f"path: ./data/SplitData\n\
train: /train/images\n\
val: /val/images\n\
test: /test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}"


f = open(f"{outputFolderPath}/data.yaml", "a")
f.write(dataYaml)
f.close()

print("Data.yaml file Created...")
