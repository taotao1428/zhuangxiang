# coding=utf-8

from dp import DP, SimpleDP
from copy import deepcopy



# 由于原生的map中，key不能为list和dict，所以写了一个简单的map，用于替代原生map
class HMap:
	def __init__(self):
		self.keys = []
		self.values = []

	def __getitem__(self, key):
		if key in self.keys:
			return self.values[self.keys.index(key)]
		return None

	def __setitem__(self, key, value):
		if key in self.keys:
			index = self.keys.index(key)
			self.values[index] = value
		else:
			self.keys.append(key)
			self.values.append(value)

	def __iter__(self):
		return iter(self.keys)

	def __len__(self):
		return len(self.keys)

	def items(self):
		l = len(self.keys)
		return [[self.keys[i], self.values[i]] for i in range(l)]

	def sort(self):
		items = self.items()
		items.sort(lambda x,y: 1 if y[1]-x[1]>0 else -1)
		self.keys = [x[0] for x in items]
		self.values = [x[1] for x in items]



'''
用于在指定多的物理机中放置虚拟机
'''
class FillDP:
	def __init__(self, compusSpec, ecssSpec, compusNum, ecssNum):
		self.compusSpec = compusSpec
		self.ecssSpec = ecssSpec
		self.ecssNum = deepcopy(ecssNum)
		self.compusNum = deepcopy(compusNum)
		self.compusEcssInfo = None
		# 按照物理机的cpu与mem的总和给物理机排序，总和小的先填
		self.compusName = sorted(compusNum.keys(), key=lambda name: compusSpec[name]['CPU'] + compusSpec[name]['MEM'])
		# print self.compusName

	# 返回放置的结果，填充情况以及剩余的虚拟机
	def getPutResult(self):
		if self.compusEcssInfo is None:
			self.startPut()
		return self.compusEcssInfo, self.ecssNum

	# 开始放置
	def startPut(self):
		self.compusEcssInfo = {}
		while not self.isEnd():
			compuName = self.nowCompuName() # 按照顺序依次拿物理机装虚拟机
			simpleDP = SimpleDP(self.compusSpec[compuName], self.ecssSpec, self.ecssNum) # 使用动态规划填充一个箱子
			count = simpleDP.getCount()
			self.updateCompusEcssInfo(count, compuName) # 将这个虚拟机的填充情况储存
			self.updateEcssNumAndCompusNum(count, compuName) # 更新ecss剩余数量与物理机剩余数量
			self.putSame(count, compuName) # 用于放置相同分布的虚拟机

	# 获取下一个需要填充的物理机
	def nowCompuName(self):
		for name in self.compusName:
			if self.compusNum[name] > 0:
				return name

	def updateCompusEcssInfo(self, count, compuName):
		if compuName not in self.compusEcssInfo:
			self.compusEcssInfo[compuName] = []
		self.lastCompu = [compuName, count] # 用于储存最后一个放置的情况，方便后面调整
		self.compusEcssInfo[compuName].append(count)

	# 判断是否所有的虚拟机已经填完或者物理机已经用完
	def isEnd(self):
		return sum(self.ecssNum.values()) == 0 or sum(self.compusNum.values()) == 0

	# 在把一个物理机放置满之后，减去已放置的虚拟机个数。同时物理机个数也减少
	def updateEcssNumAndCompusNum(self, count, compuName):
		for name in count:
			self.ecssNum[name] -= count[name]
		self.compusNum[compuName] -= 1

	# 放置一个物理机后，如果在剩余的虚拟机数量还可以放置一台这样的物理机，则进行放置
	def putSame(self, count, compuName):
		while self.hasCount(count) and self.compusNum[compuName] > 0: # 有虚拟机还要有物理机
			self.updateCompusEcssInfo(count, compuName)
			self.updateEcssNumAndCompusNum(count, compuName)

	# 查询是否满足再放置一台物理机
	def hasCount(self, count):
		for name in count:
			if self.ecssNum[name]<count[name]:
				return False
		return True


'''
在有少量虚拟机多余时，SearchBest用于搜索最好的去除方式，
'''
class SearchBest:
	def __init__(self, compusSpec, ecssSpec, compusNum, ecssNum, notFillCount, lastCompu):
		self.compusSpec = compusSpec
		self.ecssSpec = ecssSpec
		self.compusNum = compusNum
		self.ecssNum = ecssNum
		self.notFillCount = notFillCount
		self.resource = self.getNeedAdjustResource(*lastCompu) # 需要优化的资源

	def getNeedAdjustResource(self, compuName, count):
		resourceFunc = lambda resource: sum([self.ecssSpec[name][resource]*count[name] for name in count])
		rateFunc =  lambda resource: resourceFunc(resource) / float(self.compusSpec[compuName][resource])
		return 'MEM' if rateFunc('MEM') >= rateFunc('CPU') else 'CPU'

	def calcuResource(self, numsOrGroup):
		resourceCount = 0
		isNums = isinstance(numsOrGroup, dict)
		return sum([self.ecssSpec[name][self.resource]*(numsOrGroup[name] if isNums else 1) for name in numsOrGroup])

	def groupToNotFillCount(self, group):
		names = self.ecssNum.keys()
		notFillCount = {}
		for name in names:
			notFillCount[name] = 0
		for name in group:
			notFillCount[name] += 1
		return notFillCount

	# 用来检查是否需要调整
	@staticmethod
	def needAdjust(ecssSpec, ecssNum, notFillCount):
		notFillFunc = lambda resource: sum([ecssSpec[name][resource]*notFillCount[name] for name in notFillCount])
		maxFunc = lambda resource: sum([ecssSpec[name][resource] for name in ecssNum])
		return notFillFunc('CPU') <= 2*maxFunc('CPU') and notFillFunc('MEM') <= 2*maxFunc('MEM')

	def search(self):
		minResource = self.calcuResource(self.notFillCount)
		mayGroups = self.getMayGroups()
		totalResourceFunc = lambda group: sum([self.ecssSpec[name]['CPU']+self.ecssSpec[name]['MEM'] for name in group])
		names = self.ecssNum.keys()
		i = 0 # 记录周期
		while True:
			for item in mayGroups:
				item.sort(lambda x, y: totalResourceFunc(x) - totalResourceFunc(y))
				for group in item:
					group += names*i 
					self.ensureRight(group) # 确保group里面的虚拟机个数不会超
					resource = self.calcuResource(group)
					# 如果小于最小资源，说明基本不可能，所以直接跳过
					if resource < minResource:
						continue;
					newEcssNum = deepcopy(self.ecssNum)
					for name in group:
						newEcssNum[name] -= 1
					fillDP = FillDP(self.compusSpec, self.ecssSpec, self.compusNum, newEcssNum)
					compusEcssInfo, notFillCount = fillDP.getPutResult()
					if sum(notFillCount.values()) == 0:
						notFillCount = self.groupToNotFillCount(group)
						return compusEcssInfo, notFillCount
			i += 1

	def getMayGroups(self):
		names = self.ecssNum.keys()
		l = len(names)
		mayGroups = [[[name] for name in names]]
		for i in range(1, l):
			item = mayGroups[i-1]
			newItem = []
			for group in item:
				notFillCount = self.groupToNotFillCount(group) #获得现在虚拟机数量
				ii = names.index(group[-1])
				for j in range(ii+1, l):
					nextName = names[j]
					if nextName not in notFillCount:
						notFillCount[nextName] = 0
					if self.ecssNum[nextName] > notFillCount[nextName]:
						newItem.append(group + [nextName])
			mayGroups.append(newItem)
		return mayGroups

	def ensureRight(self, group):
		notFillCount = self.groupToNotFillCount(group)
		for name in notFillCount:
			if notFillCount[name] > self.ecssNum[name]:
				notFillCount[name] = self.ecssNum[name]
				i = group.index(name)
				group.pop(i)


'''
使用多种物理机放置虚拟机（复赛放置）
'''
class MultiDP:
	def __init__(self, compusSpec, ecssSpec, ecssNum):
		self.compusSpec = compusSpec
		self.ecssSpec = ecssSpec
		self.ecssNum = deepcopy(ecssNum)
		self.compusEcssInfo = None
		self.ecsCount = self.calcuEcsCount()
		self.getMayGroups()
		self.selectBestCompusEcssInfo()

	# 返回最好的填充方法以及没有填充的虚拟机
	def getBestResult(self):
		return self.bestCompusEcssInfo, self.bestNotFillCount

	def calcuEcsCount(self):
		ecsCount = {'CPU': 0., 'MEM': 0.}
		for name, num in self.ecssNum.items():
			ecsCount['CPU'] += self.ecssSpec[name]['CPU'] * num
			ecsCount['MEM'] += self.ecssSpec[name]['MEM'] * num
		return ecsCount


	# 计算某种填充方法或者某种
	def calcuVal(self, compusEcssInfoOrGroup, isInfo=False, allowRate = 1.05):
		compuCount = {'CPU': 0., 'MEM': 0.}
		for name, item in compusEcssInfoOrGroup.items():
			num = len(item) if isInfo else item
			compuCount['CPU'] += self.compusSpec[name]['CPU'] * num
			compuCount['MEM'] += self.compusSpec[name]['MEM'] * num

		rateCPU = self.ecsCount['CPU']/compuCount['CPU']
		rateMEM = self.ecsCount['MEM']/compuCount['MEM']
		val = (rateCPU+rateMEM)/2
		canFill = rateCPU<=allowRate and rateMEM<=allowRate

		if isInfo:
			return val # 如果是填充信息，仅仅返回val
		else:
			return val, canFill # 如果是计算组合的价值，还要返回是否可以将所有虚拟机全部放下

		# 计算某种填充方法或者某种
	def calcuVal2(self, compusEcssInfo, notFillCount):
		compuCount = {'CPU': 0., 'MEM': 0.}
		ecsCount = {'CPU': 0., 'MEM': 0.}
		for name, item in compusEcssInfo.items():
			num = len(item)
			compuCount['CPU'] += self.compusSpec[name]['CPU'] * num
			compuCount['MEM'] += self.compusSpec[name]['MEM'] * num

		for name, num in self.ecssNum.items():
			ecsCount['CPU'] += self.ecssSpec[name]['CPU'] * (num-notFillCount[name])
			ecsCount['MEM'] += self.ecssSpec[name]['MEM'] * (num-notFillCount[name])

		rateCPU = ecsCount['CPU']/compuCount['CPU']
		rateMEM = ecsCount['MEM']/compuCount['MEM']
		return (rateCPU+rateMEM)/2
		


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

	# 更新物理的组合，如果所有可能组合遍历完成就返回False，否者True
	def updateMayGroup(self, mayGroup, numsSingle, compusName):
		for name in compusName:
			if mayGroup[name] < numsSingle[name]:
				mayGroup[name] += 1
				return True
			else:
				mayGroup[name] = 0
		return False

	# 在所有物理机组合中选择一个最优的
	def selectBestCompusEcssInfo(self):
		bestScore = 0
		for group,originalVal in self.mayGroups.items():
			result = self.put(group)
			# 如果这种物理机组合剩余的太多，就丢弃
			if result == False:
				continue
			score, compusEcssInfo, notFillCount = result 

			print u"物理机组合：{0}".format(group)
			print u"分数：{0}".format(score)
			print u"最大的价值：%.2f, 现在的价值（填充度）：%.2f"%(originalVal, self.calcuVal2(compusEcssInfo, notFillCount))
			print u"物理机填充情况：{0}".format(compusEcssInfo)
			print u"未填充虚拟机：{0}".format(notFillCount)
			print "\n"

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

	# 计算每种放置方法的分数，最后选择一种分数最高的方法。（这个分数不是之前的价值）
	def calcuScore(self, compusEcssInfo, notFillCount):
		val = self.calcuVal2(compusEcssInfo, notFillCount)
		changeVal = sum([notFillCount[name]**2 for name in notFillCount])**0.5 \
								/ ( sum([self.ecssNum[name]**2 for name in self.ecssNum])**0.5  
										+ sum([(self.ecssNum[name]-notFillCount[name])**2 for name in self.ecssNum])**0.5)
		return (1-changeVal) * val
	

if __name__ == '__main__':
	compusSpec = {'Large-Memory'    : {'MEM': 256, 'name': 'Large-Memory', 'CPU': 84}, 
								'High-Performance': {'MEM': 192, 'name': 'High-Performance', 'CPU': 112}, 
								'General'         : {'MEM': 128, 'name': 'General', 'CPU': 56}}

	ecssSpec = {'flavor2': {'MEM': 2, 'name': 'flavor2', 'CPU': 1}, 
							'flavor1': {'MEM': 1, 'name': 'flavor1', 'CPU': 1}, 
							'flavor8': {'MEM': 8, 'name': 'flavor8', 'CPU': 4}, 
							'flavor5': {'MEM': 4, 'name': 'flavor5', 'CPU': 2}, 
							'flavor4': {'MEM': 8, 'name': 'flavor4', 'CPU': 2}}

	ecssNum = {'flavor2': 310, 'flavor1': 225, 'flavor8': 300, 'flavor5': 165, 'flavor4': 300}

	multiDP = MultiDP(compusSpec, ecssSpec, ecssNum)

	print u"最大可能价值可能大于1，因为考虑了去除一些虚拟机后的物理机组合"
	for group, val in multiDP.mayGroups.items():
		print u"{0} 组合的最大可能价值：{1}".format(group, val)

	print "\n"
	bestCompusEcssInfo, bestNotFillCount = multiDP.getBestResult()
	print u"最好的填充方法为：{0}\n".format(bestCompusEcssInfo)
	print u"会剩下的虚拟机：{0}".format(bestNotFillCount)


'''
如果运行超时可以考虑减小230行的allowRate参数（推荐保持allowRate>=1）
'''

