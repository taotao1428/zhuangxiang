# 装箱

参加[2018华为软件精英挑战赛](http://codecraft.devcloud.huaweicloud.com/home/detail)由于没有进入决赛，现在总结一下自己对初赛以及复赛的代码。题目分为预测与放置两个部分，由于在预测阶段并没有使用什么算法，所以只总结放置阶段的代码

### 多种规格虚拟机放置到一种规格物理机中，使物理机数量最小
思路：逐个物理机填充，填充的方式使用01背包。
细节：
1. 背包价值计算
```python
# 计算价值函数，等于两个维度的占比和。这样可以使物理机尽量填满
def calcuV(self, CPU, MEM):
  return CPU/float(self.compuSpec['CPU']) + MEM/float(self.compuSpec['MEM'])
```
2. 在不影响最后价值的情况下，优先体积较大的虚拟机先放
```python
# 如果产生的价值相同的情况下，优先体积大的虚拟机
# 这个函数可以控制放置过程中对虚拟机的偏好
def needReplace(self, thing, maxThing):
  return (thing['CPU'] + thing['MEM']) > (maxThing['CPU'] + maxThing['MEM'])
```
3. 放置的过程中使用一个操作类跟踪放置的过程
```python
# 操作类，用于记住每个放置操作
class Operator:
  '''
  parent: 上一个操作
  leftThings: 放置后，剩余的东西（虚拟机）
  thing:  放置的东西
  totalV： 放置后的价值
  totalCPU： 放置后CPU总数
  totalMEM： 放置后MEM总数
  '''
  def __init__(self, parent, leftThings, thing=None, totalV=0, totalCPU=0, totalMEM=0):
    self.parent = parent
    self.leftThings = leftThings
    self.thing  = thing
    self.totalV = totalV
    self.totalCPU = totalCPU
    self.totalMEM = totalMEM
  def isRoot(self):
    return self.parent is None

'''
从最后一个操作获取整个过程
'''
# 通过跟踪每个operator，获取整个放置过程放置哪些虚拟机
def countThings(self, lastOperator):
  preOperator = lastOperator
  count = {}
  while not preOperator.isRoot():
    thing = preOperator.thing
    if not thing is None:
      if not thing['name'] in count:
        count[thing['name']] = 0
      count[thing['name']] += 1
    preOperator = preOperator.parent
  return count
```
### 多种规格虚拟机放置到多种规格物理机中，使资源（CPU+MEM）利用率最高
思路：先找到多组物理机的组合方案，再进行填充，挑选出实际填充效果最好的一种组合
细节：
1. 如何找到候选物理机组合？将所有的虚拟机填入单独某一种规格的物理机中，记录**最大的利用率**（作为筛选组合的标准）和每一种**物理机的个数**（作为每一种物理机可能的最大个数）。然后再开始遍历所有可能的组合，挑选出利用率（此时计算利用率是直接采用虚拟机资源/物理机资源，并没考虑真实填充情况）会大于**单个物理机最大的利用率**的组合。
```python
def getMayGroups(self):
  # 将组合分为两组，mayFill是可能将所有的虚拟机填下去，cantFill是不可能将虚拟机填下去（需要去除部分虚拟机）
  self.mayGroups = HMap()

  # 分别获取单独使用某一种物理机装
  maxValSingle = 0 # 单个物理机最大的价值
  numsSingle = {} # 单个物理机的数量，为后面获取可能的组合提供参考
  for name, compuSpec in self.compusSpec.items():
    dp = DP(compuSpec, self.ecssSpec, self.ecssNum)
    compuEcssInfo = dp.getCompuEcssInfo()
    val = self.calcuVal({name: compuEcssInfo}, True)
    maxValSingle = max(maxValSingle, val)
    numsSingle[name] = len(compuEcssInfo)


  mayGroup = {} # 第一个组合，全部为0
  for name in self.compusSpec:
    mayGroup[name] = 0

  compusName = self.compusSpec.keys()

  while self.updateMayGroup(mayGroup, numsSingle, compusName):
    val, canFill = self.calcuVal(mayGroup)
    # 如果组合的价值大于单个物理机最好的放置
    if canFill and val >= maxValSingle:
      self.mayGroups[deepcopy(mayGroup)] = val

  self.mayGroups.sort()
```

2. 如何挑选出实际情况最好的组合？每一种组合进行填充。如果有少量虚拟机多余的话，如果直接去除多出的虚拟机可能会使预测不准，所以采用的方式是多种虚拟机分摊多出的资源，这样的话可以避免某一种虚拟机数量变化过大。然后综合变化情况与填充度计算一个score，最后挑选出一个最好的放置组合。
```python
# 在所有物理机组合中选择一个最优的
def selectBestCompusEcssInfo(self):
  bestScore = 0
  for group,originalVal in self.mayGroups.items():
    result = self.put(group)
    # 如果这种物理机组合剩余的太多，就丢弃
    if result == False:
      continue
    score, compusEcssInfo, notFillCount = result 

    if score > bestScore: # 如果获得的评分较好
      bestScore = score
      bestCompusEcssInfo = compusEcssInfo
      bestNotFillCount = notFillCount

    if sum(notFillCount.values()) == 0: # 如果当前物理机组合可以放下所有的虚拟机，后面的组合就不用考虑
      break

  self.bestCompusEcssInfo = bestCompusEcssInfo
  self.bestNotFillCount = bestNotFillCount
 
# 按照指定的物理机组合填充虚拟机
def put(self, group):
  fillDP = FillDP(self.compusSpec, self.ecssSpec, group, self.ecssNum)
  compusEcssInfo, notFillCount = fillDP.getPutResult()
  if sum(notFillCount.values()) != 0:
    # 如果有虚拟机剩余，并且剩余的太多就直接返回False（丢弃这个物理机组合）
    if not SearchBest.needAdjust(self.ecssSpec, self.ecssNum, notFillCount):
      return False
    searchBest = SearchBest(self.compusSpec, self.ecssSpec, group, self.ecssNum, notFillCount, fillDP.lastCompu)
    compusEcssInfo, notFillCount = searchBest.search() # 调整的评判标准为分
  score = self.calcuScore(compusEcssInfo, notFillCount) # 计算物理机组合最后的得分
  return score, compusEcssInfo, notFillCount

# 计算每种放置方法的分数，最后选择一种分数最高的方法。（这个分数不是之前的价值）（类似于华为比赛给我们打分的计算公式）
def calcuScore(self, compusEcssInfo, notFillCount):
  val = self.calcuVal2(compusEcssInfo, notFillCount)
  changeVal = sum([notFillCount[name]**2 for name in notFillCount])**0.5 \
              / ( sum([self.ecssNum[name]**2 for name in self.ecssNum])**0.5  
                  + sum([(self.ecssNum[name]-notFillCount[name])**2 for name in self.ecssNum])**0.5)
  return (1-changeVal) * val
```

3. 由于有多种组合需要放置，如何优化来减少计算时间？问题的核心在于01背包需要消耗大量时间。  
 - 1. 如果虚拟机资源总数可以在一个物理机中装下，就直接知道放置的方案（所有的虚拟机全部放下），这是就不需要使用dp
 - 2. 计算虚拟机中CPU与MEM的最小值作为dp循环的最小步长，这个步长会大于或等于默认步长1，这样可以加快dp的循环速度
 - 3. **对之前的dp结果进行缓存**，当存在相同情况时，可以直接从缓存中读出，这样可以极大的减少时间（这是最重要最有效的优化方案）
```python
class Cache:
  def __init__(self):
    self.cache = {}

  def compuSpecToName(self, compuSpec):
    return '[%d,%d]'%(compuSpec['CPU'], compuSpec['MEM'])

  # 判断缓存是否适合现在的情况
  def isValid(self, count, oldEcssNum, ecssNum):
    # 检查count里面有的虚拟机是必须有
    for name, num in count.items():
      if name not in ecssNum or ecssNum[name] < num:
        return False
    # 检查oldEcssNum里面没有的虚拟机，ecssNum中也需要没有
    for name, num in oldEcssNum.items():
      if num == 0 and ecssNum[name] > 0:
        return False

    return True

  def put(self, compuSpec, ecssNum, count, val):
    name = self.compuSpecToName(compuSpec)
    if name not in self.cache:
      self.cache[name] = []
    self.cache[name].append([count, val, ecssNum])
    self.cache[name].sort(lambda x,y: 1 if y[1]-x[1] > 0 else -1) # 通过使用价值排序缓存的count

  def get(self, compuSpec, ecssNum):
    name = self.compuSpecToName(compuSpec)
    if name not in self.cache:
      return None
    for count, val, oldEcssNum in self.cache[name]:
      if self.isValid(count, oldEcssNum, ecssNum):
        return count
    return None
```
