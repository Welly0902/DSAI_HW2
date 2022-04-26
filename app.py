import pandas as pd
import numpy as np
import random

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# --------------------
def getData(path):
  df = pd.read_csv(path, header=None)
  df.columns=['open','high','low','close']

  conditions = [
    (df['close']-df['open'] >= 0),
    (df['close']-df['open'] < 0)
    ]
  values = [1, 0]
  df['rise'] = np.select(conditions, values)
  df['t'] = df.close.rolling(2).mean()
  df['t'][0] = df['close'][1]
  df['diff'] = np.around(((df['close']-df['t'])/df['close'])*100)
  df['diff'][df['diff']==-0]=0
  df=df.drop(['t'],1)

  return df

def prepareTrain(df):
  df['MA5'] = df.close.rolling(5).mean()
  df['MA20'] = df.close.rolling(20).mean()
  df['MA60'] = df.close.rolling(60).mean()
  df = df[df['MA60'].notna()]
  df_train = df.copy()
  df_train['diff520'] = (df_train['MA5'] - df_train['MA20'])/df_train['MA20']
  df_train['diff560'] = (df_train['MA5'] - df_train['MA60'])/df_train['MA60']
  # df_train=df_train.drop(['MA5','MA20','MA60'],1)
  df_train.columns=['open','high','low','close','rise','diff','MA5','MA20','MA60','diff520','diff560']
  print(df_train)
  return df_train

def getLabel(df):
  df['percent'] = (df['close']-df['open'])/df['open']
  
  ## get buy time 
  # dfb = df[df['diff520']<-0.02]
  dfb = df[(df['close']<df['MA60'])&(df['close']>df['MA5'])]
  dfb2 = df[df['percent']<-0.005]
  b = list(dfb.index)
  b2 = list(dfb2.index)
  b.append(0)
  listb = getbuy(b)
  listb2 = getbuy(b2)
  for i in listb2:
    listb.append(i)

  listb.sort()

  # print(listb)
  ## get sell time
  dfs = df[df['percent']>0.005]

  s = list(dfs.index)
  s.append(0)
  lists = getsell(s)

  print(len(listb))
  print(listb)
  print(len(lists))
  print(lists)

  return listb, lists

def getbuy(l):
    retlist = []
    buylist= []
    count=1
    # Avoid IndexError for  random_list[i+1]
    for i in range(len(l) - 1):
        # Check if the next number is consecutive
        
        if l[i] + 1 == l[i+1]:
            count += 1
        else:
            # If it is not append the count and restart counting
            retlist.append(count)
            ###
            # if count > 0:
            # print(random_list[i]-count+2)
            buylist.append(l[i]-count+2)
              # df_action['action'][random_list[i]-count] = 1
            ###
            count = 1
    # Since we stopped the loop one early append the last count
    retlist.append(count)
    # return retlist[:-1]
    return buylist

def getsell(random_list):
    retlist = []
    selllist = []
    count=1
    # Avoid IndexError for  random_list[i+1]
    for i in range(len(random_list) - 1):
        # Check if the next number is consecutive
        
        if random_list[i] + 1 == random_list[i+1]:
            count += 1
        else:
            # If it is not append the count and restart counting
            retlist.append(count)
            ###
            # if count > 5:
              # print(random_list[i]-2)
            selllist.append(random_list[i]-2)
              # df_action['action'][random_list[i]-count] = 1
            ###
            count = 1
    # Since we stopped the loop one early append the last count
    retlist.append(count)
    # return retlist[:-1]
    return selllist
def pick(b,s):
  s2 = []
  b.sort(reverse=True)
  b2=b.copy()
  s.sort(reverse=True)
  s2.append(s[0])

  for fs, sc in zip(b, b[1:]):
    c=0
    for j in s:
    # print(f)
    # print(s)
    
      if (fs>j) & (sc<j):
      # print(j)
        s2.append(j)
        break
      c+=1
    # print(c)
      if c==len(s):
      # print("yo")
        b2.remove(fs)
        break

  b.sort()
  s.sort()
  b2.sort()
  s2.sort()

  return b2,s2

def mlp(n_obs=6, n_action=3, n_hidden_layer=1, n_neuron_per_layer=32,
        activation='relu', loss='mse'):
  """ A multi-layer perceptron """
  model = Sequential()
  model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
  for _ in range(n_hidden_layer):
    model.add(Dense(n_neuron_per_layer, activation=activation))
  model.add(Dense(n_action, activation='linear'))
  model.compile(loss=loss, optimizer=Adam())
  print(model.summary())
  return model

if __name__ == '__main__':
    # You should not modify this part.
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')

    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.

    ##############
    #### main ####
    ##############


    train = getData(args.training)
    train_temp = prepareTrain(train)
    # print(train.head(20))
    listb, lists = getLabel(train_temp)

    print(len(listb))
    print(len(lists))
    df2=pd.DataFrame()

    df2['long']=pd.DataFrame(np.zeros(train.shape[0]))
    df2['long'][listb]=1
    
    df2['hold']=pd.DataFrame(np.ones(train.shape[0]))
    df2['hold'][listb]=0
    df2['hold'][lists]=0

    df2['short']=pd.DataFrame(np.zeros(train.shape[0]))
    df2['short'][lists]=1

    train=train[['open','high','low','close','rise','diff']]

    test = getData(args.testing)
    # print(test)

    # print(train[train['diff']<-1.5].index)

    trainX = train
    trainY = df2
    
    # print(trainY.shape)
    # print(trainY)
    # print(trainY['long'].value_counts())
    # print(trainY['hold'].value_counts())
    # print(trainY['short'].value_counts())

    testX = test.copy()
    # trainX.shape
    # print(trainX.shape)
    # print(trainY.shape)
    # print(testX.shape)
    ans = np.zeros(testX.shape[0])


    ## prepare data for nn
    trainX=trainX.to_numpy()
    testX=testX.to_numpy()
    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # model = buildLstm(trainX)
    model = mlp()
    # model = buildModel(trainX)
    model.fit(trainX, trainY, epochs=10, verbose=1)

    ans_tmp=[]
    
    ### predict####
    p = model.predict(testX)
    print(p)
    # p[0][2]=2
    for i in p:
      # print(max(i))
      t = i.tolist()
      max_value = max(t)
      max_index = t.index(max_value)
      # print(max_index)
      ans_tmp.append(max_index)
      # print(max(t))
    # print(ans_tmp)
    ans_tmp =ans_tmp[:-1]
    # output decision
    lans = len(ans_tmp)
    if ans_tmp.count(1) == lans:
      ans_tmp[random.randint(0,lans)]=0
    else:
      actionList = []
      actionIdx = []
      for idx, val in enumerate(ans_tmp):
        if val == 0:
          actionList.append(val)
          actionIdx.append(idx)
        elif val == 2:
          actionList.append(val)
          actionIdx.append(idx) 
      # print(actionIdx)
      # print(actionList)

      c = 0

      for i in range(len(actionList)):
        if actionList[i] == 0:
          c+=1
          if c>1:
            c-=1
            actionList[i] = 1
        elif actionList[i] == 2:
          c-=1
          if c < -1:
            c+=1
            actionList[i] =1
      # print(actionList)

      for i in range(len(actionList)):
        if actionList[i]==1:
          ans_tmp[actionIdx[i]] = 1

      # print(ans_tmp)

      for i in range(len(ans_tmp)):
        ans_tmp[i] = ans_tmp[i]-1
        ans_tmp[i] = -ans_tmp[i]

      print(ans_tmp[:-1])

      df_res = pd.DataFrame(ans_tmp[:-1])
      df_res.to_csv(args.output, index=0)
      print("output.csv generated!")
