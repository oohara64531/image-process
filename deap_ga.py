#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
from deap import creator, base, tools, algorithms
import glob

#hsvフィルタ(色相:0~180,彩度:0~50,明度:100~255)
def filter_loss(input,test,individual):

    loss = 0
    img = input
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    i=0

    for gen in individual:
        if gen < 5:#[0,1,2,3,4]
            threshold = (gen+1)*10
            mask = cv2.inRange(img,(0,0,0),(180,threshold,255))
        elif gen < 10:#[5,6,7,8,9]
            threshold = (gen-4)*10
            mask = cv2.inRange(img,(0,threshold,0),(180,100,255))
        elif gen < 18:#[10,11,12,13,14,15,16,17]
            threshold = (gen-10)*20+50
            mask = cv2.inRange(img,(0,0,threshold),(180,255,255))
        elif gen < 27:#[18,19,20,21,22,23,24,25,26]
            threshold = (gen-18)*20+100
            mask = cv2.inRange(img,(0,0,50),(180,255,threshold))

        if i == 0:
            final_mask = mask
        else:
            final_mask = cv2.bitwise_and(final_mask,mask)
        i+=1

    test,rt1,rt2 = cv2.split(test)

    bitwise_xor = cv2.bitwise_xor(final_mask,test)

    loss = cv2.countNonZero(bitwise_xor)

    return loss

filter_number = list(range(27))

train_files = glob.glob('./train_data/*.jpg')
test_files = glob.glob('./test_data/*.jpg')

#適応度の最大化(weight=(-1,)で最小化)
creator.create("Fitnessmin", base.Fitness, weights=(-1.0,))
#遺伝子の定義
creator.create("Individual", list, fitness=creator.Fitnessmin)

toolbox = base.Toolbox()

filter_size = 27
ind_size = 4

#ランダムに整数をフィルタの数の間で作成
toolbox.register("attr_bool", random.sample,range(filter_size),ind_size)
#'attr_bool'で遺伝子はつくり終えているので繰り返しは一回
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool)
#遺伝子の集団を作る関数の定義
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#評価関数
def evalOneMax(individual):

    loss = 0 #lossの計算方法を考える
    num = len(train_files)
    for train_path,test_path in zip(train_files,test_files):

        train_img = cv2.imread(train_path)
        test_img = cv2.imread(test_path)
        #train画像を作成した遺伝子の番号で画像処理後にlossを出力
        tem_loss = filter_loss(train_img,test_img,individual)

        loss += tem_loss

    return loss/num,

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(64)
    #遺伝子集団の個数
    pop = toolbox.population(n=100)
    #交叉率，個体突然変異率，ループ回数
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    print("Start of evolution")
    #初期世代の適応度の計算
    fitnesses = list(map(toolbox.evaluate, pop))
    #適応度をfitness.valueに格納
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    for g in range(NGEN):
        print("-- Generation %i --" % g)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        #交叉について(偶数と奇数で確率的に交叉させる)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        #突然変異について
        for mutant in offspring:

            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        #交叉と突然変異があったので適応度の再計算
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()
