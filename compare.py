import pandas as pd

data1 = pd.read_csv('data/submission.csv')
data2=pd.read_csv('data/test_new_2.csv')
count=0
for i,data in enumerate(data1['sentiment']):
    print("data1:" + str(data1['sentiment'][i]) + " data2:" + str(data2['sentiment'][i]))
    if data1['sentiment'][i] == data2['sentiment'][i]:
        count += 1

print("正确率:"+str(count/300*100)+"%")



