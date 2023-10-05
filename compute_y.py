# Computes extremely coarse estimates of weighted days of suffering averted by
# switching from chicken to pork, as well as total suffering caused by
# eating both

def compute_y(pig_moral_weight, chicken_moral_weight):

    chickenConsPerYearKg = 57
    porkConsPerYearKg = 30

    pigWeightKg = 65
    chickenWeightKg = 2.5

    pigLifespanDays = 365
    chickenLifespanDays = 42

    y1 = chicken_moral_weight * chickenConsPerYearKg / chickenWeightKg\
        * chickenLifespanDays - pig_moral_weight * chickenConsPerYearKg\
        / pigWeightKg * pigLifespanDays

    y2 = pig_moral_weight * porkConsPerYearKg/pigWeightKg * pigLifespanDays \
        + chicken_moral_weight * chickenConsPerYearKg / chickenWeightKg \
        * chickenLifespanDays

    return y1, y2
