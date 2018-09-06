
# coding: utf-8

# In[1]:


# 以下パーセプトロンと同じ場合「*P」で記載
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 読み込み
df = pd.read_csv('file:///D:/MachineLearning/python-machine-learning/code/02/iris.data',
                header = None )
y = df.iloc[0:100 ,4 ].values
y = np.where(y == 'Iris-setosa' ,1 ,-1 )
X = df.iloc[0:100 ,[0 ,2 ] ].values
# GDの意味は知らない
class AdalineGD(object ):
# *P
    def __init__(self ,eta = 0.01 ,n_iter = 50 ,random_state = 1 ):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
# *P error_⇒cost_       
    def fit(self ,X ,y ):
        rgen = np.random.RandomState(self.random_state )
        self.w_ = rgen.normal(loc = 0.0 ,scale = 0.01 ,size = 1 + X.shape[1 ] )
        self.cost_ = []
# イメージ的には、
# Perceptronは予測値とデータとの反応を見て決定境界の傾きをずらしているが、
# Adalineは全てのデータを計算して決定境界の傾きをずらしている
        for i in range(self.n_iter ):
# 最初はランダム変数によりランダムな重みで予測値をnet_input()から持ってくる
            net_input = self.net_input(X )
# ロジスティック関数への拡張用
            output = self.activation(net_input )
# 動きはPerceptronのupdateとほぼ同じ　costに学習率を入れないように学習率は各重みの計算コードに入れてある
            errors = (y - output )
# 重みβi（すべてのデータで繰り返す）＝学習率×（xi・（yiーy_i））　y_i=予測値　※ただし、i=1,...,Nであり、バイアス項は下のコードで計算
            self.w_[1: ] += self.eta * X.T.dot(errors )
# バイアス項α＝学習率×Σ（yiーy_i）
            self.w_[0 ] += self.eta * errors.sum()
# １/2×Σ（yi-y_i）^2　経過を見る用
            cost = (errors ** 2 ).sum() / 2.0
# リストcost_にcostの結果を追加
            self.cost_.append(cost )
        return self
# *P
    def net_input(self ,X ):
        return np.dot(X ,self.w_[1: ] ) + self.w_[0 ]
# ロジスティック関数への拡張用
    def activation(self ,X ):
        return X
# *P    
    def predict(self ,X ):
        return np.where(self.activation(self.net_input(X ) ) >= 0.0 ,1 ,-1 )


# In[2]:


# 描画領域を1行2列に分割⇒左右に別々の曲線を描写　plt.subplots(行数(ｙ軸),列数(ｘ軸),サイズ（デフォルトは(8,6)）
fig ,ax = plt.subplots(nrows = 1 ,ncols = 2 ,figsize = (10 ,4 ) )
# Adalineの学習結果を代入している（学習回数10回、学習率0.01）
ada1 = AdalineGD(n_iter = 10 ,eta = 0.01 ).fit(X ,y )
# ax[0,0]⇒今回は1行なのでｙ軸を省略しているのとlog10を用いているのはサイズ感を合わせるため
ax[0].plot(range(1 ,len(ada1.cost_ ) + 1 ) ,np.log10(ada1.cost_ ) ,marker = 'o' )
# ラベルとタイトルの決定
ax[0].set_xlabel('Epochs' )
ax[0].set_ylabel('log(Sum-squared-error)' )
ax[0].set_title('Adaline - Learning rate 0.01' )
# 以下０と同様（学習回数10回、学習率0.0001）
ada2 = AdalineGD(n_iter = 10 ,eta = 0.0001 ).fit(X ,y )
ax[1].plot(range(1 ,len(ada2.cost_ ) + 1 ) ,ada2.cost_ ,marker = 'o' )
ax[1].set_xlabel('Epochs' )
ax[1].set_ylabel('Sum-squared-error' )
ax[1].set_title('Adaline - Learning rate 0.0001' )
# 学習率0.0001は回を熟すごとに修正回数が減っているが、学習率0.01は回を熟すごとに修正回数が増えている
# （修正がキツ過ぎて修正する度に修正幅が増加している）
plt.show()


# In[3]:


# データの標準化　識別をしやすくする
X_std = np.copy(X )
# (xi - μ ) / σ
X_std[: ,0 ] = (X[: ,0 ] - X[: ,0 ].mean() ) / X[: ,0 ].std()
X_std[: ,1 ] = (X[: ,1 ] - X[: ,1 ].mean() ) / X[: ,1 ].std()


# In[4]:


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
# ここでのxとyは変数のXとyとは別でx軸y軸のxとy
        plt.scatter(x = X[y == cl ,0 ] ,
                    y = X[y == cl ,1 ] ,
                    alpha = 0.8 ,
                    c = colors[idx ] ,
                    marker = markers[idx ] ,
                    label = cl ,
                    edgecolor = 'black' )


# In[13]:


# 学習回数15回、修正率0.01で、このデータではあまりお互いに近い点がないため、標準化のおかげで0.01の大胆な修正率が早い段階で識別をしてくれている
# ⇒標準化のおかげで修正数が多い学習率でも早くに高い正答率を叩き出せるが、修正数が逓減している学習率は学習回数を増やせば識別率が上がる
ada = AdalineGD(n_iter = 15 ,eta = 0.01 )
ada.fit(X_std ,y )
# adaの設定でplot_decision_regions()を起動
plot_decision_regions(X_std ,y ,classifier = ada )
# 上の図の設定
plt.title('Adaline - Gradient Descent' )
plt.xlabel('sepal length [standardized]' )
plt.ylabel('petal length [standardized]' )
plt.legend(loc = 'upper left' )
# この2行が上の図を表示　消しても上の図は残るが下の図の設定で表示され、下の図は消える
# tight_layoutは図が以下のサイズになるように設定するため
plt.tight_layout()
plt.show()
# x軸に何回目の試行か、y軸に学習回数を表示
plt.plot(range(1 ,len(ada.cost_ ) + 1 ) ,ada.cost_ ,marker = 'o' )
# 下の図の設定
plt.xlabel('Epochs' )
plt.ylabel('Sum-squared-error' )
# この2行が下の図を表示　消すとテキストとしてylabelを吐く
plt.tight_layout()
plt.show()

