import os, re, shutil, sys ,math
import xlrd
import pandas
from matplotlib import pyplot
import numpy

def read_xlsx_retirement(filename) :
    exception_words = ["emp.no","name","termination.date"] # 学習に使用しないワード
    datetime_words = ["date.of.birth","joining.date"]#,"termination.date"] # 日付のワード
    EXL = pandas.ExcelFile(filename)
    Data = EXL.parse("Sheet1")
    for word in exception_words :
        # 学習に必要のないデータの削除
        del Data[word]
    for word in datetime_words :
        # 日付をPOSIX時間に変換
        Data[word] = pandas.to_datetime(Data[word])
        Data[word] = Data[word].map(lambda x: x.timestamp())
    print(Data.columns)
    Data_dummies = pandas.get_dummies(Data) # one-hot-encoding
    print(Data_dummies.columns)
    for i in Data_dummies.columns :
        print(i)
    # 目的変数の取得と特徴量からの削除
    y = Data_dummies["retirement_Yes"]
    del Data_dummies["retirement_Yes"]
    del Data_dummies["retirement_No"]
    print(type(Data_dummies.columns))
    X = Data_dummies.values # 行列形式に変換
    print(y.shape,X.shape) # 行と列を出力
    #for i,j in zip(Data_dummies.columns,X[0]):
    #    print(i,j)
    return X,y,Data_dummies.columns

def read_xlsx_retirement2(filename) :
    exception_words = ["emp.no","name","termination.date"] # 学習に使用しないワード
    datetime_words = ["date.of.birth","joining.date"]#,"termination.date"] # 日付のワード
    string_words = ["team","biz.unit","hiring.source","sex","retirement","location","education","job.group"]
    EXL = pandas.ExcelFile(filename)
    Data = EXL.parse("Sheet1")
    for word in exception_words :
        # 学習に必要のないデータの削除
        del Data[word]
    for word in datetime_words :
        # 日付をPOSIX時間に変換
        Data[word] = pandas.to_datetime(Data[word])
        Data[word] = Data[word].map(lambda x: x.timestamp())

    convert_dict = {}
    for word in string_words :
        convert_dict[word] = {}
        counter = 0
        for s in Data[word] :
            if not s in convert_dict[word] :
                convert_dict[word][s] = counter
                counter += 1
        Data[word] = Data[word].map(convert_dict[word])
    
    print(Data.columns)
    Data_dummies = pandas.get_dummies(Data) # one-hot-encoding
    # 目的変数の取得と特徴量からの削除
    y = Data_dummies["retirement"]
    del Data_dummies["retirement"]
    X = Data_dummies.values # 行列形式に変換
    print(y.shape,X.shape) # 行と列を出力
    #for i,j in zip(Data_dummies.columns,X[0]):
    #    print(i,j)
    return X,y,Data_dummies.columns

def LinReg(X,y):
    # 線形回帰
    # y = sum w_{i}*x_{i}
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)
    clf = LinearRegression()
    fit = clf.fit(X_train, y_train)
    print("Training score : ",fit.score(X_train, y_train))
    print("Test set score : ",fit.score(X_test, y_test))
    print(fit.coef_)
    counter = 0
    for i,j in zip(y_test,fit.predict( X_test ))  :
        if abs(i-int(j>0.5)) == 1:
            print(i,j)
            counter += 1
    print("score? : %g" % (1.0 - counter/len(y_test)))
    return fit

def ridge(X,y,alpha=10.0):
    # Ridge
    # 線形回帰にL2正則化を追加したもの
    # y = sum w_{i}*x_{i} + |w|^2    
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)
    clf = Ridge(alpha=alpha)
    fit = clf.fit(X_train, y_train)
    print("Training score : ",fit.score(X_train, y_train))
    print("Test set score : ",fit.score(X_test, y_test))
    print(fit.coef_)
    counter = 0
    for i,j in zip(y_test,fit.predict( X_test ))  :
        if abs(i-int(j>0.5)) == 1:
            print(i,j)
            counter += 1
    print("score? : %g" % (1.0 - counter/len(y_test)))
    return fit

def lasso(X,y,alpha=10.0,max_iter=100000):
    # Lasso
    # 線形回帰にL1正則化を追加したもの
    # y = sum w_{i}*x_{i} + |w|
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)
    clf = Lasso(alpha=alpha,max_iter=max_iter)
    fit = clf.fit(X_train, y_train)
    print("Training score : ",fit.score(X_train, y_train))
    print("Test set score : ",fit.score(X_test, y_test))
    print(fit.coef_)
    print(fit.intercept_) # 切片
    counter = 0
    for i,j in zip(y_test,fit.predict( X_test ))  :
        if abs(i-int(j>0.5)) == 1:
            print(i,j)
            counter += 1
    print("score? : %g" % (1.0 - counter/len(y_test)))
    return fit

def LogisticReg(X,y,C=1) :
    #ロジスティック回帰(回帰とついているが分類機である)
    # y = sum w_{i}*x_{i}  + b > 0 ?
    # C は正則化のパラメータ
    # デフォルトでは、L2ノルムの項がついており、
    # 1/(2C) * sum( w_i**2 )
    # C を大きくすると正則化が弱くなる。
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(C=C)
    fit = clf.fit(X_train_scaled, y_train)
    print("Training score : ",fit.score(X_train_scaled, y_train))
    print("Test set score : ",fit.score(X_test_scaled, y_test))
    print(fit.coef_) # 係数
    print(fit.intercept_) # 切片
    #print(fit.predict_proba( X_test )) # predicの結果を確率で出力
    for i,j in zip(y_test,fit.predict( X_test_scaled ))  :
        print(i,j)
    return fit

def DecTree(X,y,max_depth=4) :
    # 決定木
    # 単純な2値問題を繰り返すことで学習を行う。
    # アルゴリズムの性質上、外挿はできない。
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)
    tree = DecisionTreeClassifier(max_depth=max_depth,random_state=0)
    fit = tree.fit(X_train,y_train)
    print("Training score : ",fit.score(X_train, y_train))
    print("Test set score : ",fit.score(X_test, y_test))
    return fit

def RandForest(X,y):
    # ランダムフォレスト
    # サンプル点をブーストラップサンプリング(復元抽出)を行い、それぞれで決定木を計算し確率予想を平均化する。
    # バカパラレルが可能。n_jobs=-1
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)
    tree = RandomForestClassifier(n_estimators=5,random_state=0)
    fit = tree.fit(X_train,y_train)
    print("Training score : ",fit.score(X_train, y_train))
    print("Test set score : ",fit.score(X_test, y_test))
    return fit

def GradBoostClassifier(X,y) :
    # 勾配ブースティング決定木
    # 決定木を計算し、間違っている部分を新たな決定木で修正する。
    # ランダムフォレストよりも時間がかかる。
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)
    tree = GradientBoostingClassifier(n_estimators=10,random_state=0)
    fit = tree.fit(X_train,y_train)
    print("Training score : ",fit.score(X_train, y_train))
    print("Test set score : ",fit.score(X_test, y_test))
    for i,j in zip(y_test,fit.decision_function( X_test ))  :
        print(i,j)
    return fit

def SVCfunc(X,y):
    # SVRは回帰、SVCは分類
    # 入力データの正規化が必要
    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    from sklearn.svm import SVC
    tree = SVC(kernel='rbf',C=10,gamma=0.1)
    fit = tree.fit(X_train_scaled,y_train)
    print("Training score : ",fit.score(X_train_scaled, y_train))
    print("Test set score : ",fit.score(X_test_scaled, y_test))
    for i,j in zip(y_test,fit.predict( X_test ))  :
        print(i,j)
    return fit

def MPLClass(X,y):
    # ニューラルネットワーク
    # 入力データの正規化が必要
    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    tree = MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10,10,10,10],activation='tanh',alpha=1.0)
    # alpha はL2正則化の係数
    fit = tree.fit(X_train_scaled,y_train)
    print("Training score : ",fit.score(X_train_scaled, y_train))
    print("Test set score : ",fit.score(X_test_scaled, y_test))
    for i,j in zip(y_test,fit.predict_proba( X_test_scaled ))  :
        print(i,j)
    return fit

def plot(X,y,keys,xvalue="OT.hour",yvalue="engagement") :
    print(X.shape)
    for i,key in enumerate(keys) :
        if key == xvalue :
            x_i = i
        if key == yvalue :
            y_i = i
    new_x = X.T[x_i]
    new_y = X.T[y_i]
    print(new_x.shape)

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i,value in enumerate(y) :
        if value == 0 :
            x1.append(new_x[i])
            y1.append(new_y[i])
        else :
            x2.append(new_x[i])
            y2.append(new_y[i])
    pyplot.plot(x1,y1,".",label="retirement No")
    pyplot.plot(x2,y2,"^",label="retirement Yes")
    pyplot.legend()
    pyplot.xlabel(xvalue)
    pyplot.ylabel(yvalue)
    #pyplot.show()
    pyplot.savefig("test.png")
    pyplot.clf()

def plot_coef_matrix(data,keys):
    # 描画する
    im = pyplot.imshow(data, interpolation='nearest')
    xtick = numpy.array(keys.values)
    locs = numpy.array(range(len(keys.values)))
    pyplot.yticks(locs, xtick, color="c", fontsize=8)
    pyplot.colorbar(im)
    #pyplot.show()
    pyplot.savefig('image.png')
    pyplot.clf()    

def plot_importances(coef,keys) :
    fig = pyplot.figure()
    fig.subplots_adjust(bottom=0.3)
    ax = fig.add_subplot(111)
    xtick = numpy.array(keys.values)
    locs = numpy.array(range(len(keys.values)))
    pyplot.xticks(locs, xtick, color="c", fontsize=8, rotation=-90)
    ax.bar(locs,coef)
    pyplot.xlim(0,len(locs))
    pyplot.savefig("figure.png")
    #pyplot.show()
    pyplot.clf()    
    
if __name__ == "__main__":
    X,y,keys = read_xlsx_retirement("kadai_data2.xlsx")

    #fit = LinReg(X,y) # 線形回帰(通常最少二乗)
    #fit = ridge(X,y,10.0) # リッジ回帰
    #fit = lasso(X,y,0.01) # Lasso
    fit = LogisticReg(X,y,1.0) # ロジスティック回帰
    coef = fit.coef_[0]

    #fit = DecTree(X,y,10) # 決定木
    #from sklearn.tree import export_graphviz
    #export_graphviz(fit,out_file="test.dot",feature_names=keys.values,impurity=False,filled=True)
    #fit = RandForest(X,y) # ランダムフォレスト
    #fit = GradBoostClassifier(X,y) # 勾配ブースティング
    #for i,j in zip(keys,fit.feature_importances_) :
    #    print(j,i)
    #coef = fit.feature_importances_
    plot_importances(coef,keys)
    
    plot(X,y,keys,xvalue="OT.hour",yvalue="engagement") 
    #fit = SVCfunc(X,y) # サポートベクターマシン # 合わなかった
    fit = MPLClass(X,y) # ニューラルネットワーク
    plot_coef_matrix(fit.coefs_[0],keys)


    # fit した結果の保存
    from sklearn.externals import joblib
    joblib.dump(fit, 'mlb_stats.pkl', compress=9) 
    # ロード
    test = joblib.load('mlb_stats.pkl')

    
