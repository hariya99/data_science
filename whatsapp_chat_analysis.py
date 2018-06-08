import subprocess as sp
tmp = sp.call('cls',shell=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import itemfreq

np.set_printoptions(threshold=np.nan)

df = pd.read_table('chat.txt')

''' Concatenating same people chats which were taken in next line(s) ''' 
bools = df.iloc[:len(df),0].str.contains(r'^\d+\/\d+\/\d+, \d+:\d+:\d+ .M:')


i = len(df) - 1
while (i >= 0):
    if bools[i] == False:
        df.iloc[i - 1, 0] += ' ' + df.iloc[i, 0]
    i -= 1

df = df[bools]  # Dropping rows whose data is appended to source row
df = df.reset_index(drop = True)    # Reformatting index 

#'''Splitting Timestamp and Chats in different columns '''
df = df.iloc[:,0].str.split('M:', 1, expand=True)
ts = df.iloc[:,0].copy()
ts = ts.reset_index(drop = True)
ts.columns = ['Timstamp']
df = df.iloc[:,1].str.split(':', 1, expand=True)
df = df.reset_index(drop = True)
df.columns = ['Name','Convo']


''' Mapping phone numbers to their person holder names '''

# for unicode use -> r'12345\xa067899'
df['Name'][df['Name'].str.contains(r'12345 45555')] = ' Friend8'
df['Name'][df['Name'].str.contains(r'98765 12222')] = ' Friend9' 


''' removing messages when someone changes their numbers '''
df = df[df['Name'].str.contains(r'\d{2} \d{5} \d{5}') == False]



'''removing rows with either of the cases:
    <image omitted>
    <video omitted>
    <GIF omitted>
'''

df = df[df['Convo'].str.contains(r'image omitted') == False]
df = df[df['Convo'].str.contains(r'video omitted') == False]
df = df[df['Convo'].str.contains(r'GIF omitted') == False]
df = df.reset_index(drop = True)



''' Counting top 3 words per friend '''

all_chat_list = []
for i in range(len(df['Name'].drop_duplicates())):
    temp = df['Convo'][df['Name'] == df['Name'].drop_duplicates().reset_index(drop = True)[i]]
    temp = temp.reset_index(drop = True)
    for j in range(1,len(temp)):
        temp[0] += ' ' + temp[j]
    all_chat_list.append(temp[0])
    del temp

fg = itemfreq(list(all_chat_list)[2].split(' '))
fg = fg[fg[:,1].astype(float).argsort()][::-1]
print(fg[1:4])

# Converting 12 hr into 24 hr logic


splitted_ts= ts.str.split(', ')
import datetime
for i in range(len(ts)):
    splitted_ts[i][1] += 'M'
    temp = datetime.datetime.strptime(splitted_ts[i][1], '%I:%M:%S %p')
    splitted_ts[i][1] = datetime.datetime.strftime(temp, '%H')

hrs = [ splitted_ts[i][1] for i in range(len(splitted_ts)) ]

hrfreq = itemfreq(hrs)
occ = [float(hrfreq[i][1]) for i in range(len(hrfreq))]
hr = [float(hrfreq[i][0]) for i in range(len(hrfreq))]

plt.plot(hr, occ)
plt.grid('on')
plt.xlabel('24 Hours')
plt.ylabel('Frequency')
plt.title('Frequent chat timings')
import subprocess as sp
tmp = sp.call('cls',shell=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import itemfreq

np.set_printoptions(threshold=np.nan)

df = pd.read_table('chat.txt')

''' Concatenating same people chats which were taken in next line(s) ''' 
bools = df.iloc[:len(df),0].str.contains(r'^\d+\/\d+\/\d+, \d+:\d+:\d+ .M:')


i = len(df) - 1
while (i >= 0):
    if bools[i] == False:
        df.iloc[i - 1, 0] += ' ' + df.iloc[i, 0]
    i -= 1

df = df[bools]  # Dropping rows whose data is appended to source row
df = df.reset_index(drop = True)    # Reformatting index 

#'''Splitting Timestamp and Chats in different columns '''
df = df.iloc[:,0].str.split('M:', 1, expand=True)
ts = df.iloc[:,0].copy()
ts = ts.reset_index(drop = True)
ts.columns = ['Timstamp']
df = df.iloc[:,1].str.split(':', 1, expand=True)
df = df.reset_index(drop = True)
df.columns = ['Name','Convo']


''' Mapping phone numbers to their person holder names '''

# for unicode use -> r'12345\xa067899'
df['Name'][df['Name'].str.contains(r'12345 45555')] = ' Friend8'
df['Name'][df['Name'].str.contains(r'98765 12222')] = ' Friend9' 


''' removing messages when someone changes their numbers '''
df = df[df['Name'].str.contains(r'\d{2} \d{5} \d{5}') == False]



'''removing rows with either of the cases:
    <image omitted>
    <video omitted>
    <GIF omitted>
'''

df = df[df['Convo'].str.contains(r'image omitted') == False]
df = df[df['Convo'].str.contains(r'video omitted') == False]
df = df[df['Convo'].str.contains(r'GIF omitted') == False]
df = df.reset_index(drop = True)



''' Counting top 3 words per friend '''

all_chat_list = []
for i in range(len(df['Name'].drop_duplicates())):
    temp = df['Convo'][df['Name'] == df['Name'].drop_duplicates().reset_index(drop = True)[i]]
    temp = temp.reset_index(drop = True)
    for j in range(1,len(temp)):
        temp[0] += ' ' + temp[j]
    all_chat_list.append(temp[0])
    del temp

fg = itemfreq(list(all_chat_list)[2].split(' '))
fg = fg[fg[:,1].astype(float).argsort()][::-1]
print(fg[1:4])

# Converting 12 hr into 24 hr logic


splitted_ts= ts.str.split(', ')
import datetime
for i in range(len(ts)):
    splitted_ts[i][1] += 'M'
    temp = datetime.datetime.strptime(splitted_ts[i][1], '%I:%M:%S %p')
    splitted_ts[i][1] = datetime.datetime.strftime(temp, '%H')

hrs = [ splitted_ts[i][1] for i in range(len(splitted_ts)) ]

hrfreq = itemfreq(hrs)
occ = [float(hrfreq[i][1]) for i in range(len(hrfreq))]
hr = [float(hrfreq[i][0]) for i in range(len(hrfreq))]

plt.plot(hr, occ)
plt.grid('on')
plt.xlabel('24 Hours')
plt.ylabel('Frequency')
plt.title('Frequent chat timings')
