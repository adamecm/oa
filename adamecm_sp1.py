import numpy as np
import pandas as pd

#Bayes rule: P(X|Y) = [P(Y|X)*P(X)]/P(Y)
#our case: P(Y) = sum(P(Xi)*P(Y|Xi))
# P(y1) = 

PX_PROBABILITY = [0.25, 0.24, 0.33, 0.18]
PYX_PROBABILITY = [0.13, 0.36, 0.2, 0.31]

#PX_PROBABILITY = [0.3, 0.22, 0.11, 0.37]
#PYX_PROBABILITY = [0.23, 0.65, 0.92, 0.5]


def bayes(x,y1,y2,y3,):
    
    return ((yx_probability(x,y1,y2,y3)* PX_PROBABILITY[x-1])/y_probability(x,y1,y2,y3))
    pass

"""
for yt in [y1,y2,y3]:
        if (yt == 0):
            full_prob *= prob_a
        else:
            full_prob *= prob_b
    return full_prob


"""

def yx_probability(x, y1, y2, y3):
  prob_a = PYX_PROBABILITY[x-1]
  prob_b = 1 - prob_a
  yx_prob = 1
  
  for yt in [y1,y2,y3]:
    if (yt == 0):
      yx_prob *= prob_a
    else:
      yx_prob *= prob_b
  return yx_prob



def y_probability(x, y1, y2, y3, n = 4):
    prob_a = PYX_PROBABILITY[x-1]
    prob_b = 1 - prob_a
    y_prob = 0
    for xi in range(1,n+1):
        y_prob += PX_PROBABILITY[xi-1]*yx_probability(xi,y1,y2,y3)
    return y_prob

result = []
order = []
for x in range(1,5):
    result.append([])
    res = []
    for y1 in range(2):
        for y2 in range(2):
            for y3 in range(2):
                tmp = bayes(x,y1,y2,y3)
                result[x-1].append(tmp)
                res.append(tmp)
                order.append((y1,y2,y3))
df = pd.DataFrame()
df = df.assign(a=order[0:8],b=result[0],c = result[1],d= result[2], e = result[3])          #[df,pd.DataFrame([np.transpose(order),np.transpose(result[0]),np.transpose(result[1]),np.transpose(result[2]), np.transpose(result[3])])])
#df.rename(columns=df.iloc[0]).drop(df.index[0])
df.columns=["Y","P(X1|Y)","P(X2|Y)","P(X3|Y)","P(X4|Y)"]
#print(df)
df2 = df[["P(X1|Y)","P(X2|Y)","P(X3|Y)","P(X4|Y)"]]
#print(df2)
argmax_prob = df2.idxmax(axis=1)
max_prob = df.max(axis=1)
print(argmax_prob)
max_prob_idx = {"P(X1|Y)":1,"P(X2|Y)":2,"P(X3|Y)":3,"P(X4|Y)":4}
#df = df.reindex(index=[7,6,5,3,4,2,1,0])
prob_correct = 0

y_prob = []
for y1 in range(2):
        for y2 in range(2):
            for y3 in range(2):
                y_prob.append(y_probability(x,y1,y2,y3))



for i in range(len(max_prob)):
    prob_correct += (max_prob[i]*y_prob[i])
print(prob_correct)
exit()
print(df)
df.to_csv("sp1.csv")
print(result[0])
print('-----------------')
print(result[1])
print('-----------------')
print(result[2])
print('-----------------')
print(result[3])

