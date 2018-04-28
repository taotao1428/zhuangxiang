# coding=utf-8
from copy import copy, deepcopy
import math
import matplotlib.pyplot as plt


class Cache:
	def __init__(self):
		self.cache = {}

	def compuSpecToName(self, compuSpec):
		return '[%d,%d]'%(compuSpec['CPU'], compuSpec['MEM'])

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

# 剩余的东西
class LeftThings:
	def __init__(self, thingsInfo):
		self.thingsInfo = thingsInfo
	# 拿走一个东西，返回新的剩余的东西
	def takeoff(self, thingName):
		thingsInfo = copy(self.thingsInfo)
		if thingsInfo[thingName] == 0:
			raise RuntimeError('没有指定的东西')
		thingsInfo[thingName] -= 1
		return LeftThings(thingsInfo)
	# 判断是否还有某件东西
	def has(self, thingName):
		return self.thingsInfo[thingName] > 0


def plotInfo(info, compu, ecss):
	getMEM = lambda name: ecss[name]['MEM']
	getCPU = lambda name: ecss[name]['CPU']
	cpuInfo = []
	memInfo = []

	for item in info:
		cpuCount = 0.
		memCount = 0.
		itemInfo = item
		for name in itemInfo:
			cpuCount += getCPU(name)*itemInfo[name]
			memCount += getMEM(name)*itemInfo[name]
		cpuInfo.append(cpuCount/compu['CPU'])
		memInfo.append(memCount/compu['MEM'])

	plt.bar( range(len(cpuInfo)),  cpuInfo, width=0.3, label='CPU')
	plt.bar( [ x+0.3 for x in range(len(memInfo))],  memInfo, width=0.3, label='MEM')
	plt.legend()
	plt.show()


# 往一个物理机里填充虚拟机
class SimpleDP:
	cache = Cache()
	def __init__(self, compuSpec, ecssSpec, ecssNum):
		self.compuSpec = compuSpec
		self.ecssSpec = ecssSpec
		self.ecssNum = deepcopy(ecssNum)
		self.count = None # 最后的结果


	def getFirstOperator(self):
		startLeftThings = LeftThings(deepcopy(self.ecssNum)) # 作为开始填充的最开始状态
		return Operator(None, startLeftThings) # 最开始的操作

	# 获取最后的结果
	def getCount(self):
		if self.count is None:
			if self.notNeedPut(): # 如果一个箱子能装满所有的虚拟机，就不需要使用动态规划去做
				self.ecssNumToCount()
			else:
				self.startPut()
		return self.count

	# 判断物理机是否可以将所有的虚拟机放下
	def notNeedPut(self):
		resourceFunc = lambda resource: sum([self.ecssSpec[name][resource]*num for name, num in self.ecssNum.items()])
		return resourceFunc('CPU') <= self.compuSpec['CPU'] and resourceFunc('MEM') <= self.compuSpec['MEM']

	def ecssNumToCount(self):
		self.count = {}
		for name, num in self.ecssNum.items():
			if num > 0:
				self.count[name] = num

	# 开始放置
	def startPut(self):
		# 尝试从缓存中获取放置的方法
		cacheCount = SimpleDP.cache.get(self.compuSpec, self.ecssNum)
		if not cacheCount is None:
			self.count = cacheCount
			return

		# 如果缓存中没有，开始放置
		minCPUStep = self.getMinStep('CPU')
		minMEMStep = self.getMinStep('MEM')

		firstOperator = self.getFirstOperator()
		maxWV = {0: {0: firstOperator}} # 用于储存最大的效率， 第一维度是CPU 第二维度是MEM
		for cpu in range(0, self.compuSpec['CPU']+1, minCPUStep):
			if not cpu in maxWV:
				maxWV[cpu] = {0: firstOperator}
			for mem in range(minMEMStep, self.compuSpec['MEM']+1, minMEMStep):
				maxV = maxWV[cpu][mem-minMEMStep].totalV # 先将最大值设为不放置任何东西
				maxThing = None
				parent = maxWV[cpu][mem-minMEMStep]
				leftThings = parent.leftThings
				for thingName in self.ecssNum:
					thing = self.ecssSpec[thingName]
					# 如果现在的cpu或者mem不足放置虚拟机，则跳过
					if mem < thing['MEM'] or cpu < thing['CPU']:
						continue
					# 获取前一步操作
					preOperator = maxWV[cpu-thing['CPU']][mem-thing['MEM']]
					preLeftThings = preOperator.leftThings
					# 检查前一步操作之后是否还剩下现在想要放置的虚拟机。如果没有了，就跳过
					if not preLeftThings.has(thingName):
						continue
					# 计算放置这个虚拟机之后的总价值
					val = self.calcuV(thing['CPU']+preOperator.totalCPU, thing['MEM']+preOperator.totalMEM)
					# 如果现在价值大于之前的价值或者满足需要替换的要求，则放置
					if val > maxV or (val == maxV and (not maxThing is None) and self.needReplace(thing, maxThing)):
						maxThing = thing
						maxV = val
						parent = preOperator
						leftThings = preLeftThings.takeoff(thingName)
				totalCPU = parent.totalCPU
				totalMEM = parent.totalMEM
				if not maxThing is None:
					totalCPU += maxThing['CPU']
					totalMEM += maxThing['MEM']
				# 储存最好的一步
				maxWV[cpu][mem] = Operator(parent, leftThings, maxThing, maxV, totalCPU, totalMEM)
		
		lastOperator = maxWV[self.compuSpec['CPU']-(self.compuSpec['CPU']%minCPUStep)][self.compuSpec['MEM']-(self.compuSpec['MEM']%minMEMStep)] # 获得最后一个操作
		self.count = self.countThings(lastOperator) # 得出最后的结果
		SimpleDP.cache.put(self.compuSpec, self.ecssNum, self.count, lastOperator.totalV) # 放到缓存中

	# 获取动态规划移动的最小值，这样可以加快速度
	def getMinStep(self, resource):
		numSet = set()
		for name, spec in self.ecssSpec.items():
			if self.ecssNum == 0: # 如果没有的虚拟机就不纳入考虑的范围
				continue
			numSet.add(spec[resource])
		return min(list(numSet))

	# 计算价值函数，等于两个维度的占比和。这样可以使物理机尽量填满
	def calcuV(self, CPU, MEM):
		return CPU/float(self.compuSpec['CPU']) + MEM/float(self.compuSpec['MEM'])

	# 如果产生的价值相同的情况下，优先体积大的虚拟机
	# 这个函数可以控制放置过程中对虚拟机的偏好
	def needReplace(self, thing, maxThing):
		return (thing['CPU'] + thing['MEM']) > (maxThing['CPU'] + maxThing['MEM'])

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




# 将所有的虚拟机放置到一种物理机中（初赛放置）
class DP:
	def __init__(self, compuSpec, ecssSpec, ecssNum):
		self.compuSpec = compuSpec
		self.ecssSpec = ecssSpec
		self.ecssNum = deepcopy(ecssNum)
		self.compuEcssInfo = None # 最后的结果

	# 获取最后的结果
	def getCompuEcssInfo(self):
		if self.compuEcssInfo is None:
			self.startPut()
		return self.compuEcssInfo

	# 开始放置
	def startPut(self):
		self.compuEcssInfo = []
		# 使用物理机装虚拟机，直到虚拟机没有
		while not self.isEnd():
			simpleDP = SimpleDP(self.compuSpec, self.ecssSpec, self.ecssNum) # 使用动态规划填充一个箱子
			count = simpleDP.getCount()
			self.compuEcssInfo.append(count) # 将这个虚拟机的填充情况储存
			self.updateEcssNum(count) # 更新ecss剩余数量
			self.putSame(count) # 用于放置相同分布的虚拟机

	# 判断是否所有的虚拟机已经填完
	def isEnd(self):
		return sum(self.ecssNum.values()) == 0

	# 在把一个物理机放置满之后，减去已放置的虚拟机个数
	def updateEcssNum(self, count):
		for name in count:
			self.ecssNum[name] -= count[name]

	# 放置一个物理机后，如果在剩余的虚拟机数量还可以放置一台这样的物理机，则进行放置
	def putSame(self, count):
		while self.hasCount(count):
			self.compuEcssInfo.append(count)
			self.updateEcssNum(count)

	# 查询是否满足再放置一台物理机
	def hasCount(self, count):
		for name in count:
			if self.ecssNum[name]<count[name]:
				return False
		return True


if __name__ == '__main__':
	# 物理机规格
	compuSpec = {'CPU': 76, 'MEM': 136}
	# 虚拟机规格
	ecssSpec = {
		'flavor1': {'name': 'flavor1', 'CPU': 1, 'MEM': 1},
		'flavor2': {'name': 'flavor2', 'CPU': 1, 'MEM': 2},
		'flavor3': {'name': 'flavor3', 'CPU': 1, 'MEM': 4},
		'flavor4': {'name': 'flavor4', 'CPU': 2, 'MEM': 2},
		'flavor5': {'name': 'flavor5', 'CPU': 2, 'MEM': 8},
		'flavor6': {'name': 'flavor6', 'CPU': 16, 'MEM': 16},
	}
	# 需要填充的物理机个数
	ecssNum = {'flavor1': 50, 'flavor2': 50, 'flavor3': 100}

	dp = DP(compuSpec, ecssSpec, ecssNum)
	compuEcssInfo = dp.getCompuEcssInfo()
	print compuEcssInfo
	plotInfo(compuEcssInfo, compuSpec, ecssSpec)