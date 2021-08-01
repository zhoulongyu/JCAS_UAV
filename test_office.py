import pandas as pd
 
data1 = []
data1.append(1234)

data1 = pd.DataFrame(data1)
writer = pd.ExcelWriter('Infor.xlsx')
data1.to_excel(writer, 'episode_rewards')
writer.save()
writer.close()