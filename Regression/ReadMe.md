### 基本形式

* 给定由d个属性描述的示例x=(x1;x2;...;xd),其中xi是x在第i个属性上的取值，线性模型试图学的一个通过属性的线性组合来进行预测的函数，即

![image](https://github.com/AngusHug/sklearn/blob/master/Regression/Image/Image1.png)
           
 w和b学的之后模型就确定了
 

* 线性回归是一种带有系数的线性模型最小化数据集中观测目标之间的残差平方和 ,并且用线性逼近法预测目标。 解决的数学问题
 ![image](https://github.com/AngusHug/sklearn/blob/master/Regression/Image/Image12.png)
               
* to minimize the residual sum of squares between the observed targets in the datasets,and targets predicted by the linear approximation
* **<u>最小二乘法是现行假设下的一种有闭式解的参数求解方法，最终结果为全局最优</u>**
    * 而梯度下降是假设条件更为广泛的，一种通过迭代更新来逐步进行的参数优化方法，最终结果为局部最优


```python
from sklearn import linear_modelreg 
res = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])    # X arraysprint(reg.coef_)
```



* <u>普通最小二乘的系数估计依赖于特征的独立性</u>。当特征相关且设计矩阵各列近似线性相关时，设计矩阵近似奇异，从而使最小二乘估计对观测目标的随机误差高度敏感，产生较大的方差。这种多重共线性的情况可能会出现，例如，在没有实验设计的情况下收集数据。
    * 奇异矩阵：行列式等于0的矩阵。
    * <u> 可决系数:表示一个随机变量与多个随机变量关系的数字特征，用来反映回归模式说明因变量变化可靠程度的一个指标。可以定义为：已被模式中全部自变量说明的自变量的变差对自变量总变差的比值</u>

* 通常如果有多个w解，都能使均方误差最小化。选择哪个解作为输出将由学习算法的归纳偏好决定，常见的做法是引入<u>正则项</u> ======》主要为了限制模型多重共线性w的增长(也是常见的防止过拟合方法)，在模型原来的目标函数加上一个惩罚项(求解线性回归的解，对最小二乘法的变形,优化最小二乘法)。模型对输入w中噪声的敏感度就会降低。===》脊回归和套索回归的基本思想
    * 如果惩罚项是L1正则化，就是Ridge回归，岭回归。
    * 如果使用的是L2正则化，就是Lasso回归
    
![image](https://github.com/AngusHug/sklearn/blob/master/Regression/Image/Image2.png)

    * L1正则化是指权值向量w中各个元素的绝对值之和，通常表示为||w||1。
        *  L1正则化可以使得参数稀疏化，即得到的参数是一个稀疏矩阵，可以用于特征选择。
        * 稀疏性：就是模型的很多参数是0.通常机器学习中特征向量很多。在预测或分类时，特征难以选择，但是如果带入这些特征得到的模型是一个稀疏模型，很多参数是0，表示只有少数特征对这个模型有贡献，绝大多数特则会那个没有贡献，即使去掉对模型也没有影响，此时就只关注系数是非零值的特征。相当于对模型进行了一次特则会给你选择，只留下一些重要特征，提高模型繁华的能力，降低过拟合的可能。
    * L2正则化是指权值向量w中各个元素的平方和后然后再求平方根
    * 一般会在正则化项之前添加一个系数lambda。这个系数需要用户指定，就是我们要调的超参。
[参考链接](https://www.cnblogs.com/zingp/p/10375691.html)
#### Ridge regression

    * 脊回归通过添加惩罚项解决最小二乘法的问题
    
![image](https://github.com/AngusHug/sklearn/blob/master/Regression/Image/Image3.png)

    * 脊回归和一般线性回归的区别是在损失函数上增加了一个L2的正则化项，和一个调节线性回归项和正则化项全中的系数alpha.

```python
reg.intercept
```

*     RidgeClassifier:脊回归的分类器变量。
    * This classifier first converts binary targets to {-1, 1} and then treats the problem as a regression task, optimizing the same objective as above. 
    * For multiclass classification, the problem is treated as multi-output regression, and the predicted class corresponds to the output with the highest value

##### 设置正则化参数：通过交叉验证的方法

* RidgeCV使用内置的alpha参数交叉验证实现回归。RidgeCV的工作方式与GridSearchCV相同，只是它默认使用通用交叉验证(GCV)，这是一种有效的遗漏交叉验证形式

```python
import numpy as np
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.logspace(-6, -6,13))
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

reg.alpha
```

#### Lasso

* lasso是一个估计稀疏系数的线性模型。因为它倾向于选择具有较少非零系数的解，有效地减少了给定解所依赖的特征的数量。因此Lasso及其变体在压缩感知领域极其重要。在一定的条件下，它可以恢复非零系数的精确集合。<u>由于Lasso回归得到的是稀疏模型，因此可以用它来进行特征选择，详见基于l1的特征选择。</u>
* 数学意义：带有正则项的线性模型

 ![image](https://github.com/AngusHug/sklearn/blob/master/Regression/Image/Image4.png)

* Lasso添加了alpha||w||1来解决求解最小二乘法的问题。其中alpha是一个常数 ||w||1是一个l1范数的稀疏向量。
* 使用坐标下降法作为算法拟合系数

```python
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]],[0, 1])
res.predict([[1, 1]])
```
##### 设置惩罚项参数

* 参数控制估计系数的稀疏程度

##### 交叉验证

* 对于Lasso通过交叉验证获取alpha ：LassoCV、LassoLarsCV.LassoLarsCV都是基于最小角回归法

* 对于具有许多共线特征的高维数据集：
    * LassoCV通常是最可取的。
    * LassoLarsCV的优势在于探索更多相关的alpha参数值
    * 如果与特征的数量相比，样本的数量非常少，那么LassoLarsCV通常比LassoCV要快。

* 与支持向量机的正则化参数比较：支持向量机正则化参数C与的等价性由= 1 / C或= 1 / (n_samples * C)给出，具体取决于估计量和模型优化的精确目标函数
* 多任务Lasso
    * 数学意义：它由一个混合训练的线性模型组成为正则化准则。最小化i的目标函数
 ![image](https://github.com/AngusHug/sklearn/blob/master/Regression/Image/Image5.png)
    
    
#### Elastic-Net

* ElasticNet是一个线性回归模型，同时使用L1和L2作为正则项。这种结合可以让你像Lasso那样从一个稀疏模型中学到少量非零权重，也能想Ridge模型那样仍然维持正则化属性。使用l1_ratio参数控制l1和l2的凸结合
* ElasticNet适用于多个相互关联的复杂特征。Lasso可能会选取其中一个，但是elastic-net会选取相关的两个。
* 在Lasso和Ridge之间进行平衡的实际好处是使得elastic-net可以继承Ridge在rotation上的稳定性。
* Elastic-Net的最小化目标函数

![image](https://github.com/AngusHug/sklearn/blob/master/Regression/Image/Image6.png)

    * ElasticNetCV可以通过加查验证来设置参数alpha和l1_ratio
##### 多任务Elastic-Net

* MultiTaskElasticNet是一种预估多元回归的稀疏系数的线性模型。Y是(n_samples,n_task)组合的二维数组。所选择的约束条件对于所有回归问题都是一致的，也称为task。
* 数据意义上表达为L1和L2组成的混合训练线性模型正则惩罚项最小化目标函数
![image](https://github.com/AngusHug/sklearn/blob/master/Regression/Image/Image7.png)
#### 最小角回归

* 最小角回归是一个针对高维数据的回归算法。最小角回归和正向逐步回归相似。每一步，他都找到和目标相关性最高的特征。当多个特征具有相等的相关性时，不是沿着同一特征继续而是沿着特征间的等角方向继续
* 优势
    * 在特征数明显大于样本数的情况下，该方法在数值上是有效的。
    * 它的计算速度与正向选择一样快，复杂度与普通最小二乘法相同。
    * 它生成完整的分段线性解决方案路径，这在交叉验证或类似的模型调优尝试中非常有用。
    * 如果两个特征与目标的相关性几乎相等，那么它们的系数应该以近似相同的速度增加。因此，该算法的性能与直觉预期一致，而且更稳定。

* 劣势
    * 由于LARS是基于残差反复修正迭代，它似乎对噪音的影响特别敏感
##### LARA Lasso

LassoLars是一个使用LARS算法实现的lasso模型，与基于坐标下降的实现不同，它生成精确的解，该解是其系数范数的分段线性函数。

#### Orthogonal Matching Pursuit （OMP 正交匹配追踪）

OrthogonalMatchingPursuit或orthogonal_mp实现了OMP算法，该算法通过对非零系数(即系数)的数量施加约束来逼近线性模型的拟合。

正交匹配追踪是一种类似于最小角度回归的正向特征选择方法，它可以用固定数量的非零元素逼近最优解向量

### Bayesian Regression
贝叶斯回归可以在估计过程中包含正则化参数。正则化参数没有严格设置，根据手头数据调整。
这可以通过在模型的超参数上引入不提供信息的先验来实现。<u>岭回归和分类中使用的正则化等价于在高斯先验条件下对系数进行精确的最大后验估计。不需要手动设置lambda，可以将其作为一个随机变量，从数据中进行估计。</u>

为了得到一个全概率模型，假设输出是高斯分布的:
![image](https://github.com/AngusHug/sklearn/blob/master/Regression/Image/Image8.png)
#### 贝叶斯岭回归

BayesianRidge估计了上述回归问题的概率模型。系数的先验由球面高斯分布给出:
![image](https://github.com/AngusHug/sklearn/blob/master/Regression/Image/Image9.png)

先验函数被选择为伽马分布，它是高斯函数精度的共轭先验函数。得到的模型被称为贝叶斯岭回归。在模型拟合过程中，参数与正则化参数共同估计，并通过最大对数边际似然估计(maximizing the log marginal likelihood

#### 自动相关性测定 ARD)

ARD与贝叶斯海脊回归非常相似，但可能导致更稀疏的系数w[1][2]。通过放弃高斯分布是球形的假设，ARD提出了一个不同的先验.相反的ARD分布被认为是一个轴向平行的椭圆高斯分布。
这意味着每个系数都是高斯分布的，以零为中心，具有精度lambda。

![image](https://github.com/AngusHug/sklearn/blob/master/Regression/Image/Image10.png)





### Logistic regression逻辑回归

逻辑回归，尽管它的名字，是一个线性模型的分类，而不是回归。
Logistic回归在文献中也被称为logit回归、最大熵分类(MaxEnt)或对数线性分类器。在这个模型中，描述单个试验可能结果的概率用逻辑函数来建模。
LogisticRegression实现了逻辑回归，此时先可以选择L1,L2或者Elastic-Net正则化拟合二进制，
One-vs-Rest或多项逻辑回归。
![image](https://github.com/AngusHug/sklearn/blob/master/Regression/Image/Image11.png)





* 剪枝是决策树学习算法对付过拟合的主要手段。有时结点划分过程不断重复，会造成决策树分支过多，这是就可能因为训练样本学的太火，以至于把训练集自身的一些特点当做所有数据都具有的一般性质而导致过拟合。
* 剪枝的基本策略有**预剪枝**和**后剪枝**

    *    **预剪枝**：在决策树生成过程中，对每个结点在划分前先进行预估，若当前结点的划分不能带来决策树泛化性能提升，则停止划分并将当前节点标记为叶结点
    *    **后剪枝**：先从训练集生成一颗完整的决策树，然后自底向上的对非叶结点进行考察，若将该结点对应的子树替换为叶结点能带来决策树泛化性能提升，则将该子树替换为叶结点。

##### 多变量决策树
若吧每个属性视为坐标空间中的一个坐标轴，则d个属性描述的样本就对应了d维空间中的一个数据点，对样本分类意味着在这个坐标空间中寻找不同类样本之间的分类边界。

* **决策树形成的分类边界有一个明显的特点：轴平行，及它的分类边界由若干个与坐标轴平行的分段组成。**
