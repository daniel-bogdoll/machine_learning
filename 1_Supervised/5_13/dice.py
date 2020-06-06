def bayesTheorem(pA, pRA, pB, pRB):
    pAR = (pA * pRA) / (pA * pRA + pB * pRB)
    pBR = (pB * pRB) / (pA * pRA + pB * pRB)
    return [pAR, pRB]

dice_normal = [1,2,3,4,5,6]
dice_weird = [2,3,3,4,4,5]

#5 dice in bag
p_dice_normal = 3/5
p_dice_weird = 2/5

p_dice_normal_3 = 1/6
p_dice_weird_3 = 1/3

p_dice_rolled_is_standard =  bayesTheorem(p_dice_normal, p_dice_normal_3, p_dice_weird, p_dice_weird_3)
print(p_dice_rolled_is_standard)
