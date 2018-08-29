
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ネットから拾ってきた通称アヤメデータを参照する
df = pd.read_csv('file:///D:/MachineLearning/python-machine-learning/code/02/iris.data',
                 header=None )
# ４列目の１～１００行目のデータをyに代入
y = df.iloc[0:100 ,4 ].values
# yに入っている’Iris-setosa’をー１に、それ以外を１に変換
y = np.where(y == 'Iris-setosa' ,-1 ,1 )
# １～２列目の１～１００行目のデータをXに代入
X = df.iloc[0:100 ,[0 ,2 ] ].values
# x軸にXの１列目、ｙ軸にXの２列目のデータを１～５０行目と５１～１００行目で色を分けてプロット
plt.scatter(X[:50 ,0 ] ,X[:50 ,1 ] ,color = 'red' ,marker = 's' ,label = 'setosa' )
plt.scatter(X[50:100 ,0 ] ,X[50:100 ,1 ] ,color = 'blue' ,marker = 'v' ,label = 'versicolor' )
# ｘ軸とｙ軸のラベルを設定
plt.xlabel('sepal length [cm]' )
plt.ylabel('petal length [cm]' )
# データのラベルの位置を設定
plt.legend(loc = 'upper left' )

print(y )

plt.show()


# In[14]:


class Perceptron(object):
# eta（学習率）を0.01、n_iter（学習回数）を50、random_stateを1として初期化 
    def __init__(self ,eta = 0.01 ,n_iter = 50 ,random_state = 1 ):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
# rgenをrandom_stateをシードとしてランダム変数を生成するインスタンㇲ、w_を平均0、分散0.01、（Xの次元＋バイアス項）行のベクトル、errors_をリストとして何も入力せずに定義
    def fit(self ,X ,y ):
        rgen = np.random.RandomState(self.random_state )
        self.w_ = rgen.normal(loc = 0.0 ,scale = 0.01 ,size = 1 + X.shape[1 ] )
        self.errors_ = []
# n_iter回、errorsを初期化しながらXとyの行数分（次元分）以下の学習を行う
        for _ in range(self.n_iter ):
            errors = 0
# xiにXの値を、targetにyの値を入れながらXとyの少ない方のデータ数分試行を繰り替えす（Xとyのデータ数は同じはず）
# updateにeta×（yの値ーpredict関数にxiを代入して得た値）を代入
# w_[0]にはバイアス項としてupdateの累積を蓄え、他のw_にはupdateにxiを掛けて修正
# errorsにupdateが試行回数とpredict関数の値が異なったとき（predict関数は下の説明のとおりyの予測値を変えす）、１を加える
            for xi,target in zip(X ,y ):
                update = self.eta * ( target - self.predict(xi ) )
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0 )
# リストのerrors_に変数errorsを加える
            self.errors_.append(errors )
        return self
# Xとw_のバイアス項を除いた項を内積し、バイアス項を加えた値を返す関数を定義（α+βxi、予測値）
    def net_input(self ,X ):
        return np.dot(X ,self.w_[1: ] ) + self.w_[0 ]
# net_input関数の値が０か正の時１を、負の時ー１を返す（閾値関数）
def predict(self ,X ):
        return np.where(self.net_input(X ) >= 0.0 ,1 ,-1 )

# ppnを学習率0.1、学習回数10回のパーセプトロンを呼ぶ変数として定義
ppn = Perceptron(eta = 0.1 ,n_iter = 10 )
# Xとyでppnの条件の下、学習をする 
ppn.fit(X ,y )
# x軸に1からスタートしてppnのerrors_を、y軸に回ごとのerrors_の数値（update回数）を与えてグラフ化
plt.plot(range(1 ,len(ppn.errors_ ) + 1 ) ,ppn.errors_ ,marker = '.' )
plt.xlabel('Epochs' )
plt.ylabel('Number of update' )
plt.show()
# 一応各数値を確認
print(ppn.w_[0: ] ,ppn.errors_ )


# In[20]:


from matplotlib.colors import ListedColormap
def plot_decision_regions(X ,y ,classifier ,resolution = 0.02 ):
# colorsの中からｙの識別対象としているクラス文左から当てはめる
    markers = ('s' ,'x' ,'o' ,'^' ,'v' )
    colors = ('red' ,'blue' ,'lightgreen' ,'gray' ,'cyan' )
    cmap = ListedColormap(colors[:len(np.unique(y ) ) ] )
#  x1(2)_minにXの最小の値ー１を、x1(2)_maxにXの最大の値＋１を与える
    x1_min ,x1_max = X[: ,0 ].min() - 1 ,X[: ,0 ].max() + 1
    x2_min ,x2_max = X[: ,1 ].min() - 1 ,X[: ,1 ].max() + 1
# xxiは行列で、行にx1の最小から最大まで0.02毎に値を与え、x2の長さ分行または列にコピー（x2に対する操作は行と列、１と２を交換して）
    xx1 ,xx2 = np.meshgrid(np.arange(x1_min ,x1_max,resolution ) ,
                           np.arange(x2_min ,x2_max,resolution ) )
# xx1、xx2を1行にして(xxi.ravel())、1列目にxx1を、2列目にxx2を入れた（np.array().T←転置）行列でpredict関数を行うようにzを設定
    z = classifier.predict(np.array([xx1.ravel() ,xx2.ravel() ] ).T )
# zに入っていた値をxx1の要素数を1行の制限として改行
    z = z.reshape(xx1.shape )
# x軸にxx1を、y軸にxx2を指定し、cmapの設定の下、zの高さで等高線を引く
# alphaは０～１であらわす透明度
    plt.contourf(xx1 ,xx2 ,z ,alpha = 0.3 ,cmap = cmap )
# x軸もy軸もxx1とxx2がすべて描けるような範囲で描写
    plt.xlim(xx1.min(),xx1.max() )
    plt.ylim(xx2.min(),xx2.max() )
# 各点を図にプロット
    for idx,cl in enumerate(np.unique(y ) ):
        plt.scatter(x = X[y == cl ,0 ] ,
                    y = X[y == cl ,1 ] ,
                    alpha = 0.8 ,
                    c = colors[idx ] ,
                    marker = markers[idx ] ,
                    label = cl ,
                    edgecolor = 'black' )


# In[21]:


# 決定境界をプロット
plot_decision_regions(X ,y ,classifier = ppn )
plt.xlabel('sepal length [cm]' )
plt.ylabel('petal length [cm]' )
plt.legend(loc = 'upper left' )
plt.show()

