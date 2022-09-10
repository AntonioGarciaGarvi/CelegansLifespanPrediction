import re
import math
from functools import reduce

class CData:
    def __init__(self, name):
        self.name = name
        self.timelist = []
        self.deadlist = {} # dead objects
        self.censoredlist = {} # censored objects
        self.survpercent = []
        self.survpercent_CI95 = []
        self.timelist_mot = []
        self.mortality = []
        self.mortality_CI95 = []
        self.total = 0
        self.all = 0
        self.survall = 0

        ## Calculate for plot
        self.km_data_list = []

        self.km_s_list = []  # The Kaplan-Meier estimator is the nonparametric maximum likelihood estimate of S(t). (from wikipedia)
        self.km_v_list = []  # Greenwood's formula for approximate variance of Kaplan-Meier estimator. (from wikipedia)
        self.ci_list = [] # Confidence interval of a survival curve; http://www.graphpad.com/www/Book/survive.htm

        self.mean = 0.0
        self.stderr = 0.0
        self.low_ci = 0.0
        self.high_ci = 0.0
        self.ci = 0.0
        self.at_25 = "-"
        self.at_50 = "-"
        self.at_75 = "-"
        self.at_90 = "-"
        self.at_100 = "-"
        self.median_ci = "-"

    def GetSurvData(self):
        time = []
        status = []
        
        for day in self.deadlist:
            for i in range(0, self.deadlist[day]):
                time.append( day )
                status.append( 1 ) # dead
        
        for day in self.censoredlist:
            for i in range(0, self.censoredlist[day]):
                time.append( day )
                status.append( 0 ) # censored

        return time, status

class KMData:
    def __init__(self):
        pass
    
def CalculateSubplotData( conditions ):
    for i in range( 0, len( conditions ) ):
        aData1 = conditions[i]
        cut_list = [ 25, 50, 75, 90, 100, 110 ]
        next_cut = 25.0
        prev_percent = 0.0
        percent = 0 #aData1.survall
        MEAN = 0
        S = 1.0
        prev_S = 1.0
        Vtemp = 0.0
        SURV_TIME = 0.0
        SE = 0.0
        Risk = aData1.all
        for index in aData1.timelist:
            if Risk != 0:
                Old_S = S
                S *= ( 1 - aData1.deadlist[index] / float(Risk) )  
                if Risk - aData1.deadlist[index] != 0:
                    Vtemp += ( aData1.deadlist[index] / ( float(Risk) * ( float(Risk) - aData1.deadlist[index] ) ) ) 
                else:
                    Vtemp = 0
                V = S * S * Vtemp
                SURV_TIME += ( (Old_S - S ) * index )
            aKMData = KMData()
            aKMData.time = index
            aKMData.at_risk = Risk
            aKMData.dead = aData1.deadlist[ index ]
            aKMData.censored = aData1.censoredlist[ index ]
            aKMData.percent_mortality = (1.0 - S) * 100.0 
            aKMData.s_hat = S
            aKMData.var_s_hat = V
            aKMData.se_s = math.sqrt( V )
            aKMData.surv_time = SURV_TIME
            if prev_percent < next_cut <= aKMData.percent_mortality:
                next_cut += 25.0
                aKMData.bold = True
            else:
                aKMData.bold = False 
            prev_percent = aKMData.percent_mortality

            aData1.km_data_list.append( aKMData )

            survpercent = S 
            aData1.survpercent.append( survpercent * 100 ) ## For Survival plot
            if ( survpercent != 0.0 ):
                a = ( - math.log( survpercent ) ) ## For Survival plot : Mortality Rate
                if ( a != 0.0 ):
                    b = math.log( a ) ## Log cumulative hazard plot
                    aData1.timelist_mot.append( index ) ## For Survival plot
                    aData1.mortality.append( b ) ## For Survival plot : Mortality Rate Curve (? )
            SE += math.sqrt(V)

            Risk -= aData1.deadlist[index]
            Risk -= aData1.censoredlist[index]
            prev_S = S
            
def GetData( name, conditions ):
    for aData in conditions:
        if aData.name == name:
            return aData
    aData = CData( name )
    conditions.append( aData )
    return aData

def KM_STATISTICS( aData1 ):
    V = 0.0
    M = 0.0
    S_list = []
    S_dic = {}
    S = 1.0
    old_index = 0
    Risk = aData1.all
    for index in aData1.timelist:
        if Risk != 0:
            Old_S = S
            S_list.append( S )
            S_dic[index] = S
            S *= ( 1 - aData1.deadlist[index] / float(Risk) )  # multiply ( 1 - di/ni )
            M += ( Old_S * ( index - old_index ) )            # Mean survival time is estimated as the area under the survival curve.
            old_index = index
        Risk -= aData1.deadlist[index]
        Risk -= aData1.censoredlist[index]

    Risk = aData1.all
    for i in range( 0, len(aData1.timelist) ):
        old_index = aData1.timelist[ i ]
        ST = 0.0
        for j in range( i, len(aData1.timelist) ):
            index = aData1.timelist[ j ]
            ST += ( S_dic.get( index, 0 ) * ( index - old_index ) )            # Mean survival time is estimated as the area under the survival curve.
            old_index = index

        index = aData1.timelist[ i ]
        di = float( aData1.deadlist[index] )
        if Risk != 0 and (Risk - di) != 0 :
            V += ( float(ST) * float(ST) * di / ( float(Risk) * ( float(Risk) - di ) ) )
        Risk -= aData1.deadlist[ index ]
        Risk -= aData1.censoredlist[ index ]
 
    return [M, math.sqrt(V), 1.959963985*math.sqrt(V)]  # SURV_TIME MEAN, VARIANCE, 95% CONFIDENCE INTERVAL


def Statistics( conditions ):
    for aData in conditions:
        cut_list = [ 25, 50, 75, 90, 100, 110 ]
        next_cut = 0
        percent = 0 #aData1.survall
        MEAN = 0
        S = 1.0
        Vtemp = 0.0
        SURV_TIME = 0.0
        SE = 0.0
        Risk = aData.all
        median_cis = []
        BC_Value = 0
        for index in aData.timelist:
            if Risk != 0:
                Old_S = S
                S *= ( 1 - aData.deadlist[index] / float(Risk) )  # multiply ( 1 - di/ni )
                if Risk - aData.deadlist[index] != 0:
                    Vtemp += ( aData.deadlist[index] / ( float(Risk) * ( float(Risk) - aData.deadlist[index] ) ) ) # sum ( di / (ni*(ni-di)) )
                else:
                    Vtemp = 0
                V = S * S * Vtemp
                SURV_TIME += ( (Old_S - S ) * index )
            percent = 1.0 - S
            if  cut_list[next_cut] <= percent*100:
                if next_cut == 0:
                    aData.at_25 = "%d" % index
                if next_cut == 1:
                    aData.at_50 = "%d" % index
                if next_cut == 2:
                    aData.at_75 = "%d" % index
                if next_cut == 3:
                    aData.at_90 = "%d" % index
                if next_cut == 4:
                    aData.at_100 = "%d" % index
            try:
                BC_Value = ( S - 0.5 ) / math.sqrt( V )
                if -1.96 <= BC_Value <= 1.96:
                    median_cis.append( index )
                pass
            except:
                pass
            SE += math.sqrt(V)
            if cut_list[next_cut] <= percent * 100:
                next_cut += 1

            Risk -= aData.deadlist[index]
            Risk -= aData.censoredlist[index]
        try:
            aData.median_ci = "%s ~ %s" % ( median_cis[0], median_cis[-1] )
        except:
            aData.median_ci = "%s ~ %s" % ( "-", "-" )

        [ aData.mean, aData.stderr, aData.ci ] = KM_STATISTICS( aData )
        aData.low_ci = aData.mean - aData.ci
        aData.high_ci = aData.mean + aData.ci

        
## READ DATA
def ReadData( textData, conditions ):
    aData = None
    for line in textData:
        line = line.strip()
        if len( line ) == 0:
            continue

        if line[0] == "#":
            continue
        elif line[0] == "%":
            name = line[1:]
            search_result = re.search( "\[(?P<total>\d+)\]", name )

            if search_result != None:
                name = name[:search_result.start()].strip()
                aData = GetData( name, conditions ) 
                aData.total += int( search_result.group( "total" ) )
            else:
                aData = GetData( name, conditions )
        else:
            fields = []
            for value in line.split("\t"):
                value = value.strip()
                if value == "":
                    fields.append( 0 )
                else:
                    fields.append(int(float(value))) ###### modificado
                    # fields.append( int( value ) )

            day = int( fields[0] )            ## Observed Time

            if ( len( fields ) < 2 ):         ## Number of dead subjects
                dead_cnt = 0
            else:
                dead_cnt = int( fields[1] )

            if ( len( fields ) <= 2 ):        ## Number of missing subjects for censoring
                censored_cnt = 0
            else:
                censored_cnt = int( reduce( lambda x, y: int(x)+int(y), fields[2:] ) )
              
            #if dead_cnt == 0 and censored_cnt == 0: ## No data
            #    continue

            if day not in aData.timelist:
                aData.timelist.append( day )

            aData.deadlist[ day ] = aData.deadlist.get( day, 0 ) + dead_cnt
            aData.censoredlist[ day ] = aData.censoredlist.get( day, 0 ) + censored_cnt

            aData.all += dead_cnt
            aData.survall += dead_cnt
            aData.all += censored_cnt

            if search_result == None:
                aData.total += dead_cnt
                aData.total += censored_cnt

    for aData in conditions:
        aData.timelist.sort()
        
    for aData in conditions:
        if aData.all == aData.total or aData.total == 0:
            continue
        day_last = max( aData.deadlist.keys() )
        aData.censoredlist[ day_last ] += ( aData.total - aData.all )
        aData.all = aData.total

def ReadDataFromFile( filepath ):
    f = open( filepath )
    data = f.readlines()
    f.close()
    
    conditions = []
    ReadData( data, conditions )

    return conditions

def load_sample_data():    
    testTxt = '''% WT [52]\t\t
#days\tdead\tcensored
3\t0\t
6\t0\t6
8\t1\t4
12\t9\t3
15\t8\t1
18\t6\t
22\t3\t
25\t2\t
28\t5\t
30\t3\t
32\t1\t
\t\t

% daf-2(-) [223]\t\t
#days\tdead\tcensored
3\t1\t3
6\t0\t20
12\t1\t40
15\t7\t36
17\t2\t5
21\t3\t6
24\t2\t2
26\t7\t
29\t15\t
31\t7\t
34\t15\t
36\t18\t
39\t17\t
41\t13\t3
\t\t

% daf-16(-) [127]\t\t
#days\tdead\tcensored
3\t0\t5
6\t2\t7
8\t1\t10
12\t11\t6
15\t29\t
18\t28\t
22\t15\t
25\t13\t
28\t\t
30\t\t
32\t\t
\t\t

% daf-16(-); daf-2(-) [366]\t\t
#days\tdead\tcensored
3\t0\t7
6\t10\t50
12\t13\t55
15\t58\t53
17\t23\t4
21\t28\t8
24\t22\t2
26\t10\t
29\t13\t
31\t3\t
34\t3\t
36\t2\t
39\t2\t
41\t\t
'''
    conditions = []
    ReadData( testTxt.split( "\n" ), conditions )

    return conditions


# if __name__ == "__main__":
#     conditions = load_sample_data()
#
#     for day in conditions[0].timelist:
#         print (day, conditions[0].deadlist[day], conditions[0].censoredlist[day])
