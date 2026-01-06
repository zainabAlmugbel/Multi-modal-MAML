import csv
import pdb
img_file_path="/mnt/scratch/users/zha503/Unicorn Folder/data/cub/CUB_200_2011/images.txt"
train_id="/mnt/scratch/users/zha503/Unicorn Folder/data/cub/trainclasses.txt"
valid_id="/mnt/scratch/users/zha503/Unicorn Folder/data/cub/valclasses.txt"
test_id="/mnt/scratch/users/zha503/Unicorn Folder/data/cub/testclasses.txt"

tntid=[]
valid=[]
tstid=[]
with open(train_id, 'r') as file:
    # Read one line at a time
    tntid = [line.strip() for line in file.readlines()]


with open(valid_id, 'r') as file:
    # Read one line at a time
    valid = [line.strip() for line in file.readlines()]
        
    
with open(test_id, 'r') as file:
    # Read one line at a time
    tstid = [line.strip() for line in file.readlines()]


with open(img_file_path, 'r') as file:
    # Read one line at a time
    imgs = [line.strip() for line in file.readlines()]
    
tn_matches=[]
for tn_id in tntid:
    for img in imgs:
        if tn_id in img:
            tn_matches.append(img)# = [text for text in imgs if tn_id in text]

val_matches=[]
for val_id in valid:
    for img in imgs:
        if val_id in img:
            val_matches.append(img)

ts_matches=[]
for tst_id in tstid:
    for img in imgs:
        if tst_id in img:
            ts_matches.append(img)
tn_data=[]
val_data=[]
ts_data=[]

for v in tn_matches:
    img_path=v.split(" ")[1]
    clss_img=img_path.split(".")[0]
    row= [img_path ,clss_img]
    tn_data.append(row)

for v in val_matches:
    img_path=v.split(" ")[1]
    clss_img=img_path.split(".")[0]
    row= [img_path ,clss_img]
    val_data.append(row)


for v in ts_matches:
    img_path=v.split(" ")[1]
    clss_img=img_path.split(".")[0]
    row= [img_path ,clss_img]
    ts_data.append(row)

with open("train.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(tn_data)

with open("test.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(ts_data)

with open("val.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(val_data)
#print(tntid)

#print(valid)
#print(tstid)
