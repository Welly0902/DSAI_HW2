

import pandas as pd
import numpy as np



from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

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
  ## get buy time
  dfb = df[df['diff520']<-0.03]
  b = list(dfb.index)
  b.append(0)
  listb = getbuy(b)
  # print(listb)
  ## get sell time
  dfs = df[(df['MA5']>df['MA20']) & (df['MA20']>df['MA60'])]
  
  s = list(dfs.index)
  s.append(0)
  lists = getsell(s)
  # print(lists)

  listb, lists = pick(listb,lists)

  print(len(listb))
  print(listb)
  print(len(lists))
  print(lists)
  # df['action']= np.zeros(df.shape[0])
  # df['action'][listb]=1
  # df['action'][lists]=-1
  # df[(df['action']==1) | (df['action']==-1)]
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
            if count > 5:
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

def buildLstm(df):
  # model = Sequential()
  # model.add(LSTM(30, input_length=shape[1], input_dim=shape[2]))
  # # output shape: (1, 1)
  # model.add(Dense(1))
  # model.compile(loss="mse", optimizer="adam")
  # model.summary()

  n_features = df.shape[1]

  # define model
  model = Sequential()
  model.add(LSTM(50, activation='relu', input_shape=(1, n_features)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  # model.fit(x_train, y_train, epochs=1000, verbose=1)
  model.summary()
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
    train['action']= np.zeros(train.shape[0])
    train['action'][listb]=1
    train['action'][lists]=-1
    train[(train['action']==1) | (train['action']==-1)]
    train=train[['open','high','low','close','rise','diff','action']]
    print(train[(train['action']==1) | (train['action']==-1)])

    test = getData(args.testing)
    print(test)

    trainX = train.drop('action', 1)

    trainY = train['action']
    testX = test.copy()
    # trainX.shape
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = buildLstm(trainX)
    model.fit(trainX, trainY, epochs=300, verbose=1)
    # train_data = data[:, :3526]
    # test_data = data[:, 3526:]
    # print(train_data)
    # # print(data)

    # env = TradingEnv(train_data, 1000)
    # state_size = env.observation_space.shape
    # print(state_size)
    # action_size = env.action_space.n
    # print(action_size)
    # agent = DQNAgent(state_size, action_size)
    # print(agent)
    # scaler = get_scaler(env)
    # print(scaler)

    # portfolio_value = []

    # if args.mode == 'test':
    #   # remake the env with test data
    #   env = TradingEnv(test_data, args.initial_invest)
    #   # load trained weights
    #   agent.load(args.weights)
    #   # when test, the timestamp is same as time when weights was trained
    #   timestamp = re.findall(r'\d{12}', args.weights)[0]

    # # for e in range(args.episode):
    # for e in range(5):
    #   state = env.reset()
    #   state = scaler.transform([state])
    #   for time in range(env.n_step):
    #     action = agent.act(state)
    #     next_state, reward, done, info = env.step(action)
    #     next_state = scaler.transform([next_state])
    #     if args.mode == 'train':
    #       agent.remember(state, action, reward, next_state, done)
    #     state = next_state
    #     if done:
    #       print("episode: {}/{}, episode end value: {}".format(
    #         e + 1, 5, info['cur_val']))
    #       portfolio_value.append(info['cur_val']) # append episode end portfolio value
    #       break
    #     if args.mode == 'train' and len(agent.memory) > 32:
    #     # if args.mode == 'train':
    #       agent.replay(32)
    #   if args.mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
    #     agent.save('weights/{}-dqn.h5'.format(timestamp))

    # # save portfolio value history to disk
    # with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
    #   pickle.dump(portfolio_value, fp)

# ----
    # training_data = load_data(args.training)
    # trader = Trader()
    # trader.train(training_data)
    
    # testing_data = load_data(args.testing)
    # with open(args.output, 'w') as output_file:
    #     for row in testing_data:
    #         # We will perform your action as the open price in the next day.
    #         action = trader.predict_action(row)
    #         output_file.write(action)


    #         # this is your option, you can leave it empty.
    #         trader.re_training(i)