#Computes extremely coarse estimates of weighted days of suffering averted by 
#switching from chicken to pork, as well as total suffering caused by 
#eating both

def compute_y(pigMoralWeight,chickenMoralWeight):
    
    chickenConsPerYearKg = 57
    porkConsPerYearKg = 30
    
    pigWeightKg = 65
    chickenWeightKg = 1.9
    
    pigLifespanDays = 183
    chickenLifespanDays = 42
    
    y1 = chickenMoralWeight*chickenConsPerYearKg/chickenWeightKg*chickenLifespanDays-\
        pigMoralWeight*chickenConsPerYearKg/pigWeightKg*pigLifespanDays \
        
    y2 = pigMoralWeight*porkConsPerYearKg/pigWeightKg*pigLifespanDays \
        + chickenMoralWeight*chickenConsPerYearKg/chickenWeightKg*chickenLifespanDays
    
    return y1, y2