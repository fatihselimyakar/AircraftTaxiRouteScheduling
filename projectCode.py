################################################
# Author: Fatih Selim YAKAR                    #
# Optimization Term Project                    #
# Instructor: Prof. Dr. Fatih Erdogan SEVILGEN #
# Aircraft Taxiing                             #
# Used Python Version: Python 3.7              #
################################################

import sys
import math
import random
import datetime
import time

MAX_INT =2147483647

# Holds the flights spesifications like a C structure
class Flight():
    # Contructor of Flight
    def __init__(self):
        self.flightNo=""
        self.etd=0.
        self.type=0
        self.priority=0
        self.stand=0
        self.isDeparture=False
        self.source=0
        self.destination=0

    # Setter of Flight
    def setFlight(self,flightNo,etd,typeIp,priority,stand,isDeparture,source,destination):
        self.flightNo=flightNo
        self.etd=etd
        self.type=typeIp
        self.priority=priority
        self.stand=stand
        self.isDeparture=isDeparture
        self.source=source
        self.destination=destination

    # Getter of Flight
    def getFlight(self):
        return [self.flightNo,self.etd,self.type,self.priority,self.stand,self.isDeparture,self.source,self.destination]


# Type of adjacency matrix graph class
class Graph():
    # Constructor of class
    def __init__(self,size):
        tempArray=[0]*size
        self.graphMatrix=[]
        for i in range(size):
            self.graphMatrix.append(tempArray.copy())

        self.size=size
        self.range=0

    # Gets the edge's weight between vertex1 and vertex2
    def getEdgeWeight(self,vertex1,vertex2):
        return self.graphMatrix[vertex1][vertex2]

    # Add a new edge in graph
    def addEdge(self,vertex1,vertex2,weight):
        self.graphMatrix[vertex1][vertex2]=weight
        self.graphMatrix[vertex2][vertex1]=weight
        #print(self.graphMatrix[vertex2][vertex1])

    # Delete the vertex1-vertex2 edge
    def deleteEdge(self,vertex1,vertex2):
        self.graphMatrix[vertex1][vertex2]=0
        self.graphMatrix[vertex2][vertex1]=0

    # getAllPaths's helper
    def getAllPathsUtil(self, u, d, visited, path,pathArray): 
        visited[u]= True
        path.append(u) 
        if u == d: 
            self.range-=1
            pathArray.append(path.copy())
        else: 
            for i in range(len(self.graphMatrix[u])): 
                if visited[i]== False and self.graphMatrix[u][i]!=0: 
                    self.getAllPathsUtil(i, d, visited, path,pathArray)
                if(self.range==0):
                    return None 
                      
        path.pop() 
        visited[u]= False
   
   
    # Gets all paths from 's' to 'd' 
    def getAllPaths(self, s, d): 
        self.range=1000
        visited =[False]*(self.size) 
        path = [] 
        pathArray=[]
        self.getAllPathsUtil(s, d, visited, path,pathArray) 
        return pathArray

# Specifies the all algorithm's needed features
class Algorithm():
    # Constructor of Algorithm, initializes the graph and flights
    def __init__(self):
        self.graph=Graph(44)
        self.graph.addEdge(2,12,9)
        self.graph.addEdge(3,13,8)
        self.graph.addEdge(4,14,8)
        self.graph.addEdge(5,15,8)
        self.graph.addEdge(6,16,8)
        self.graph.addEdge(7,17,8)
        self.graph.addEdge(8,18,8)
        self.graph.addEdge(9,19,9)
    
        self.graph.addEdge(10,11,4)
        self.graph.addEdge(10,33,4)
        self.graph.addEdge(33,32,4)
        self.graph.addEdge(33,38,5)
        self.graph.addEdge(38,40,4)

        self.graph.addEdge(11,12,8.5)
        self.graph.addEdge(11,32,4)
        self.graph.addEdge(32,31,8.5)
        self.graph.addEdge(32,40,5)
        self.graph.addEdge(40,41,8.5)

        self.graph.addEdge(12,13,4.5)
        self.graph.addEdge(12,31,4)
        self.graph.addEdge(31,30,4.5)
        self.graph.addEdge(31,41,5)
        self.graph.addEdge(41,37,11.37)
        
        self.graph.addEdge(13,14,7.5)
        self.graph.addEdge(13,30,4)
        self.graph.addEdge(30,29,7.5)
        self.graph.addEdge(30,37,8.5)
        self.graph.addEdge(37,34,7.5)

        self.graph.addEdge(14,15,7.5)
        self.graph.addEdge(14,29,4)
        self.graph.addEdge(29,28,7.5)
        self.graph.addEdge(29,34,8.5)
        self.graph.addEdge(34,27,8.5)
        self.graph.addEdge(34,35,7.5)

        self.graph.addEdge(15,16,7.5)
        self.graph.addEdge(15,28,4)
        self.graph.addEdge(28,27,7.5)
        self.graph.addEdge(28,35,8.5)
        self.graph.addEdge(35,26,8.5)
        self.graph.addEdge(35,36,8.5)

        self.graph.addEdge(16,17,7.5)
        self.graph.addEdge(16,27,4)
        self.graph.addEdge(27,26,7.5)

        self.graph.addEdge(17,18,7.5)
        self.graph.addEdge(17,26,4)
        self.graph.addEdge(26,25,7.5)

        self.graph.addEdge(18,19,7.5)
        self.graph.addEdge(18,25,4)
        self.graph.addEdge(25,24,7.5)
        self.graph.addEdge(25,36,8.5)
        self.graph.addEdge(36,42,14.37)

        self.graph.addEdge(19,20,7.5)
        self.graph.addEdge(19,24,4)
        self.graph.addEdge(24,23,7.5)
        self.graph.addEdge(24,42,5)
        self.graph.addEdge(42,43,7.5)

        self.graph.addEdge(20,21,4)
        self.graph.addEdge(20,23,4)
        self.graph.addEdge(23,22,4)
        self.graph.addEdge(23,43,5)
        self.graph.addEdge(43,39,4)

        self.graph.addEdge(21,22,4)
        self.graph.addEdge(22,39,5)

        self.flights=[]
        for i in range(9): self.flights.append(Flight())
        self.flights[0].setFlight("MU5178",0,320,4,2,True,2,38)
        self.flights[1].setFlight("CZ3118",0,330,4.5,2,True,2,38)
        self.flights[2].setFlight("CZ6218",0,330,4.5,3,True,3,38)
        self.flights[3].setFlight("MU2078",0,320,4,4,True,4,38)
        self.flights[4].setFlight("CA1605",0,737,5,3,True,3,38)
        self.flights[5].setFlight("CA1802",0,738,3.5,2,False,37,2)
        self.flights[6].setFlight("MF8115",0,737,3,4,False,37,4)
        self.flights[7].setFlight("HU7196",0,734,2.5,6,False,37,5)
        self.flights[8].setFlight("GS6574",0,319,2,5,False,37,5)

        # path pool for choicing flight paths
        self.paths=[]
        for i in range(len(self.flights)):
            self.paths.append(self.graph.getAllPaths(self.flights[i].getFlight()[-2],self.flights[i].getFlight()[-1]))


        self.unit=75
        self.minimumSafetyTime=30
        self.minimumSafetyLength=300
        self.aircraftSpeed=36

    # Returns the given parameter path's length
    def pathLength(self,path):
        totalLength=0
        for i in range(len(path)-1):
            totalLength+=self.graph.getEdgeWeight(path[i],path[i+1])

        return totalLength

    # Transforms pathList to pathArray (a -1 b to [a,b] )
    def returnPathArray(self,pathList):
        beginIndex=0
        pathArray=[]
        for i in range(len(pathList)):
            if(pathList[i]==-1):
                pathArray.append(pathList[beginIndex:i].copy())
                beginIndex=i+1
        return pathArray

    # Transforms pathArray to pathList ([a,b] to a -1 b )
    def pathArrayToPathList(self,pathArray):
        pathList=[]
        for i in range(len(pathArray)):
            pathList.extend(pathArray[i])
            pathList.append(-1)
        return pathList

    # Calculates the objective function of taxiing problem
    def objFuncFind(self,pathArray):
        speed=10
        
        timedPaths=[]
        for i in range(len(pathArray)):
            pathZero=[]
            for j in range(len(pathArray[i])):
                if(j==0):
                    pathZero.append([pathArray[i][j],self.flights[j].etd])
                else:
                    time=(self.graph.getEdgeWeight(pathArray[i][j-1],pathArray[i][j])*self.unit)/speed
                    pathZero.append([pathArray[i][j],pathZero[j-1][1]+time])        
            timedPaths.append(pathZero)

        maxCounter=0
        while(self.nodeConflictDetect(timedPaths)):
            for i in range(len(pathArray)):
                self.nodeConflictFix(timedPaths,i)

            for i in range(len(pathArray)):
                self.edgeConflictFix(timedPaths,i)
            if(maxCounter==100):
                break
            maxCounter+=1

        for i in range(len(pathArray)):
            self.edgeConflictFix(timedPaths,i)

        objValue=0
        for lst in timedPaths:
            objValue+=lst[-1][1]
        
        return timedPaths,objValue

    # Increases all the times after the index
    def increaseTime(self,path,index,time):
        for i in range(index,len(path)):
            path[i][1]+=time

        return path

    # Fixes the node conlict
    def nodeConflictFix(self,pathArray,index):
        path=pathArray[index]
        for i in range(len(path)):
            for j in range(len(pathArray)):
                for k in range(len(pathArray[j])):
                    if(j!=index and pathArray[j][k][0]==path[i][0]):
                        if(math.fabs(pathArray[j][k][1]-path[i][1])<self.minimumSafetyTime):
                                incTime=0
                                if(self.flights[index].priority>self.flights[j].priority):
                                    incTime=self.minimumSafetyTime+path[i][1]-pathArray[j][k][1]
                                    self.increaseTime(pathArray[j],k,incTime)
                                else:
                                    incTime=self.minimumSafetyTime+pathArray[j][k][1]-path[i][1]
                                    self.increaseTime(path,i,incTime)
        return pathArray


    # Fixes the edge conflict
    def edgeConflictFix(self,pathArray,index):
        path=pathArray[index]
        # For traversing main array's edge
        for i in range(len(path)-1):
            for j in range(len(pathArray)):
                for k in range(len(pathArray[j])-1):
                    if(j!=index):
                        if(path[i][0]==pathArray[j][k+1][0] and path[i+1][0]==pathArray[j][k][0]):
                            if((path[i+1][1]>pathArray[j][k][1] and pathArray[j][k][1]>=path[i][1]) or (pathArray[j][k+1][1]>path[i][1] and path[i][1]>=pathArray[j][k][1])):
                                incTime=0
                                if(self.flights[index].priority>self.flights[j].priority):
                                    incTime=path[i+1][1]-pathArray[j][k][1]+self.minimumSafetyTime
                                    self.increaseTime(pathArray[j],k,incTime)
                                else:
                                    incTime=pathArray[j][k+1][1]-path[i][1]+self.minimumSafetyTime
                                    self.increaseTime(path,i,incTime)

    # Detects the node conflict
    def nodeConflictDetect(self,pathArray):
        nodeArray=[]
        for i in range(44):
            nodeArray.append([])

        for i in range(len(pathArray)):
            for j in range(len(pathArray[i])):
                nodeArray[pathArray[i][j][0]].append(pathArray[i][j][1])

        for i in range(len(nodeArray)):
            if(nodeArray[i]!=None):
                for j in range(len(nodeArray[i])):
                    for k in range(len(nodeArray[i])):
                        if(k!=j):
                            if(math.fabs(nodeArray[i][j]-nodeArray[i][k])<self.minimumSafetyTime):
                                return True

        return False

    # GA CODES BELOW

    # Random population generates for the GA
    def randomPopulationGeneration(self,size):

        population=[]
        for i in range(size):
            innerPath=[]
            for j in range(len(self.paths)):
                innerPath.append(self.paths[j][random.randint(0,len(self.paths[j])-1)])
            population.append(self.pathArrayToPathList(innerPath))
        
        return population

    # Select the new chromosomes for the GA
    def rouletteWheelSelection(self,fitnessArray,totalFitnessValue):
        fitnessArraySize=len(fitnessArray)
        upperBound=0
        lowerBound=0
        rand=random.random()*totalFitnessValue

        for i in range(0,fitnessArraySize):
            if(i==0):
                upperBound=fitnessArray[0]
                lowerBound=0
            else:
                lowerBound+=fitnessArray[i-1]
                upperBound+=fitnessArray[i]
            if(rand>=lowerBound and rand<=upperBound):
                return i
        return i

    # Selects the mating pool for the GA
    def matingPoolSelection(self,genes):
        populationSize=len(genes)
        fitnessArray=[]
        totalFitnessValue=0
        totalUpdatedFitness=0
        matingPool=[]
        for i in range(populationSize):
            fitnessArray.append(self.objFuncFind(self.returnPathArray(genes[i]))[1])
            totalFitnessValue+=fitnessArray[i]
        
        for i in range(0,populationSize):
            fitnessArray[i]=totalFitnessValue/fitnessArray[i]
            totalUpdatedFitness+=fitnessArray[i]
        
        #print(fitnessArray,totalUpdatedFitness)
        for i in range(0,populationSize):
            selection=self.rouletteWheelSelection(fitnessArray,totalUpdatedFitness)
            matingPool.append(genes[selection].copy())

        return matingPool

    # Applies the crossover process for two parameter chromosomes
    def xoverProcess(self,chromosome1,chromosome2):
        xoverLine=random.randint(1,len(self.flights)-1)
        pathArray1=self.returnPathArray(chromosome1)
        pathArray2=self.returnPathArray(chromosome2)
        for i in range(xoverLine):
            pathArray1[i],pathArray2[i]=pathArray2[i],pathArray1[i]
        
        return self.pathArrayToPathList(pathArray1),self.pathArrayToPathList(pathArray2)

    # Applies the crossover for all of the population
    def xover(self,population,xoverProb):
        i=0
        while(i<len(population)):
            if(random.random()<=xoverProb):
                population[i],population[i+1]=self.xoverProcess(population[i],population[i+1])
            i+=2

    # Applies the mutation
    def mutation(self,population):
        for i in range(len(population)):
            pathArray=self.returnPathArray(population[i])
            randForPath=random.randint(0,len(pathArray)-1)
            randForNewPath=random.randint(0,len(self.paths[randForPath])-1)
            pathArray[randForPath]=self.paths[randForPath][randForNewPath].copy()
            population[i]=self.pathArrayToPathList(pathArray)

    # Survivor selection function, selects the survivors comparing the old and new ones
    def survivorSelection(self,population,newPopulation):
        for i in range(len(population)):
            popValue=self.objFuncFind(self.returnPathArray(population[i]))[1]
            popNewValue=self.objFuncFind(self.returnPathArray(newPopulation[i]))[1]
            if(popNewValue<popValue):
                population[i]=newPopulation[i].copy()

    # Finds best objective function value in population
    def findBestInPopulation(self,population):
        bestValue=float('inf')
        bestPath=[]
        avgValue=0
        for i in range(len(population)):
            value=self.objFuncFind(self.returnPathArray(population[i]))
            if(value[1]<bestValue):
                bestValue=value[1]
                bestPath=population[i]
            avgValue+=value[1]

        return bestValue,bestPath,avgValue/len(population)

    # Applies the Genetic Algorithm
    def GA(self,populationSize,xoverProb,loopSize):
        bestValue=float('inf')
        bestPath=[]
        genes=self.randomPopulationGeneration(populationSize)
        for i in range(loopSize):
            matingPool=self.matingPoolSelection(genes)
            self.xover(matingPool,xoverProb)
            self.mutation(matingPool)
            localBest=self.findBestInPopulation(matingPool)
            if(localBest[0]<bestValue):
                bestValue=localBest[0]
                bestPath=localBest[1]
            self.survivorSelection(genes,matingPool)
        return bestValue

    # SA CODES BELOW

    # Simulated Annealing's neighbourhood function
    def neighbourhoodSA(self,pathList):
        path=self.returnPathArray(pathList)
        randInt=random.randint(0,len(path)-1)
        randInt2=random.randint(0,len(self.paths[randInt])-1)
        path[randInt]=self.paths[randInt][randInt2]

        return self.pathArrayToPathList(path)

    # Creates the initial solution for the SA
    def createInitialSolution(self):
        path=[]
        for j in range(len(self.paths)):
            path.append(self.paths[j][random.randint(0,len(self.paths[j])-1)])

        return self.pathArrayToPathList(path)

    # Applies the SA algorithm
    def SA(self,temperatureCooling,innerLoop,coolestTemp,reheatSize):
        bestTeta=float('inf')
        bestSolution=[]
        initialSolution=self.createInitialSolution()
        reheatTemperature=self.objFuncFind(self.returnPathArray(initialSolution))[1]
        fTeta=reheatTemperature
        for rh in range(reheatSize):
            counter=0
            temperature=reheatTemperature
            while(temperature>=coolestTemp):
                i=0
                while(i<innerLoop):
                    newSolution=initialSolution.copy()
                    newSolution=self.neighbourhoodSA(newSolution)
                    fNewTeta=self.objFuncFind(self.returnPathArray(newSolution))[1]
                    if((fNewTeta-fTeta<=0) or (random.random()<=math.exp(-((fNewTeta-fTeta)/temperature)))):
                        fTeta=fNewTeta
                        initialSolution=newSolution.copy()
                    if(fTeta<bestTeta):
                        bestTeta=fTeta
                        bestSolution=initialSolution.copy()
                    i+=1
                #print(bestTeta,fTeta)
                temperature*=temperatureCooling
                counter+=1
            reheatTemperature*=0.9
            #print(bestTeta)
        return bestTeta,bestSolution

# TEST FUNCTIONS BELOW

# Tests the cooling coefficient
def testFunctionSaCooling():
    alg=Algorithm()
    coolingRatio=0.1
    resArray=[]
    reheatArray=[21,14,12,9,6,5,4,2,1]
    for j in range(9):
        subResArray=[]
        start_time = time.time()
        for i in range(10):
            subResArray.append(alg.SA(coolingRatio,20,0.05,reheatArray[j])[0])
        print(" %s seconds for cooling ratio:" % (time.time() - start_time),coolingRatio)
        coolingRatio+=0.1
        resArray.append(sum(subResArray)/len(subResArray))
    
    print("Cooling Ratio 0.1 to 0.9, coolestTemp:0.05, inner loop size:20")
    print(resArray)

# Tests the SA inner loop
def testFunctionSaInnerLoop():
    alg=Algorithm()
    innerLoop=10
    resArray=[]
    reheatArray=[10,5,4,3,2]

    for j in range(5):
        subResArray=[]
        start_time = time.time()
        for i in range(10):
            subResArray.append(alg.SA(0.9,innerLoop,0.05,reheatArray[j])[0])
        print(" %s seconds for inner loop" % (time.time() - start_time),innerLoop)
        innerLoop+=10
        resArray.append(sum(subResArray)/len(subResArray))

# Tests the number of reheat for the SA
def testFunctionSaReheat():
    alg=Algorithm()
    for i in range(10):
        print("SA finished:",alg.SA(0.6,20,800)[0])

# Tests the population size of the GA
def testFunctionGaPopulationSize():
    alg=Algorithm()
    popSize=[10,20,30,40]
    for j in range(4):
        print("popSize:",popSize[j]," Crossover prob:",0.8,"loopSize:",int(1000/popSize[j]))
        totalRes=0
        for i in range(10):
            totalRes+=alg.GA(popSize[j],0.8,int(2000/popSize[j]))
        print("Result:",totalRes/10)
        
# Tests the crossover probability of the GA
def testFunctionGaCrossoverProb():
    alg=Algorithm()
    crossOverProb=0.5
    for j in range(5):
        print("popSize:",10," Crossover prob:",crossOverProb,"loopSize:",100)
        totalRes=0
        for i in range(10):
            totalRes+=alg.GA(10,crossOverProb,100)
        print("Result:",totalRes/10)
        crossOverProb+=0.1

# Tests the crossover probability of the GA
def testFunctionGaIteration():
    for i in range(10):
        array=[]
        totalRes+=alg.GA(10,0.9,400)
    
    print("Result:",totalRes/10)

if __name__ == "__main__":
    start_time = time.time()
    random.seed(datetime.time())
    print("Program started")
    alg=Algorithm()
    alg2=Algorithm()
    print("Result of the GA:",alg2.GA(10,0.9,400))
    print("Result of the SA:",alg.SA(0.9,50,0.05,6)[0])
    print("--- Program finished in %s seconds ---" % (time.time() - start_time))





