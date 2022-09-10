'''
% Log Rank Test
% This program provides statistical tests for the comparison of two survival functions through overall lifespan data.
%
% This program requires the statlib.
%
%    statlib ==> http://code.google.com/p/python-statlib/
%
% Last Update: 2011.06.03
% By Jae-Seong Yang, destine@postech.ac.kr
'''

import SurvivalData
from scipy.stats import chi2

def LogRankTest( D1, D2 ):

    N1j = float( D1.all )
    N2j = float( D2.all )

    V_Chi = 0.0
    D_Chi = 0.0

    days = set( D1.timelist )
    days.update( D2.timelist )

    days = list( days )
    days.sort() 

    for j in days:
        Nj = float( N1j + N2j )

        if ( Nj == 0.0 ):
            continue


        O1j = D1.deadlist.get( j, 0 )
        O2j = D2.deadlist.get( j, 0 )

        D = ( O1j + O2j )
        E = N1j * D / Nj

        D_Chi += ( O1j - E )
        if ( Nj == 1 ):
            V_Chi += 0 
        else:
            V_Chi += ( N1j * N2j * D * ( Nj - D ) ) / ( Nj ** 2 * ( Nj - 1 ) ) 

        N1j -= D1.deadlist.get( j, 0 )
        N2j -= D2.deadlist.get( j, 0 )
        N1j -= D1.censoredlist.get( j, 0 )
        N2j -= D2.censoredlist.get( j, 0 )


    X_Chi = ( D_Chi ** 2 ) / V_Chi
    P_value = chi2.sf(X_Chi, 1) # Original-> P_value = stats.lchisqprob( X_Chi, 1 )
    
    return [ D1.name, D2.name, X_Chi, P_value ]

def LogRankTestAll( conditions ):
    results_logRankTest = []

    for i in range( 0, len( conditions ) ):
        aData1 = conditions[i]
        for j in range( 0, len( conditions ) ):
            if i == j:
                continue
            aData2 = conditions[j]

            [ name1, name2, X_Chi, P_value ] = LogRankTest( aData1, aData2 )
            Corrected_P_value = min( P_value * ( len( conditions ) - 1 ), 1.0 )
            
            results_logRankTest.append( [ name1, name2, X_Chi, P_value, Corrected_P_value ] )

    return results_logRankTest

#
# if __name__ == "__main__":
#     conditions = SurvivalData.ReadDataFromFile( "sample.txt" )
#
#     results_logRankTest = LogRankTestAll( conditions )
#
#     print("\t".join( ["Sample 1", "Sample 2", "Chi^2", "P-value", "Corrected P-value"] ))
#     for results in results_logRankTest:
#         print(results)
#         # for v in results:
#         #     print(v, "\t",)
#         print("")
