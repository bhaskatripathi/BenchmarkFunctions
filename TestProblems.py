# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 00:45:04 2020

@author: Bhaskar Tripathi
"""

import numpy  as np
import math


'''Beale'''
def F1(x):
    x = np.asarray_chkfinite(x)
    x1 = x[0]
    x2 = x[1]
    s = ((1.5 - x1 + x1 * x2) ** 2.0 + (2.25 - x1 + x1 * x2 ** 2.0) ** 2.0+ (2.625 - x1 + x1 * x2 ** 3.0) ** 2.0)
    return s

'''Easom'''    
def F2(x):
    x = np.asarray_chkfinite(x)
    term1=-np.cos(x[0])    
    term2=np.cos(x[1]) 
    term3=np.exp(-1*((np.float_(x[0])-np.pi)**2 + (np.float_(x[1]) - np.pi) ** 2)) 
    s=term1*term2*term3    
    return s

'''Matyas'''
def F3(x):
    #y=0.26*(x(1)^2+x(2)^2)-0.48*x(1)*x(2); 
    x = np.asarray_chkfinite(x)
    x1 = x[0]
    x2 = x[1]
    s = 0.26 * (np.power(x1,2.0) + np.power(x2,2)) - 0.48 * np.multiply(x1,x2)
    #s= 0.26*(np.sum(np.power(x1,2),np.power(x2,2)))-0.48*np.multiply(x1,x2) 
    return s

'''Powell'''
def F4(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    n4 = ((n + 3) // 4) * 4
    if n < n4:
        x = np.append( x, np.zeros( n4 - n ))
    x = x.reshape(( 4, -1 ))  # 4 rows: x[4i-3] [4i-2] [4i-1] [4i]
    term = np.empty_like( x )
    term[0] = x[0] + 10 * x[1]
    term[1] = np.sqrt(5) * (x[2] - x[3])
    term[2] = np.power((x[1] - 2 * x[2]),2)
    term[3] = np.sqrt(10) * np.power((x[0] - x[3]),2)
    return np.sum( term**2 )

''' Commenting this as it is not Powell function'''
#def F4(x):
#    o=np.max(np.abs(x))
#    return o
    
    
'''Schaffer No.1'''    
def F5(x):
    x = np.asarray_chkfinite(x)
    x1 = x[0]
    x2 = x[1]
    s = 0.5 + ((np.power(np.sin(np.power(x1,2) - np.power(x2,2)),2) - 0.5))/ (1 + (0.001 * np.power(np.power(x1,2)+ np.power(x2,2),2)))
    return s

'''Schaffer No. 3'''    
def F6(x):
    x = np.asarray_chkfinite(x)
    x1 = x[0]
    x2 = x[1]
    term1 = np.power(np.sin(np.cos(np.power(np.abs(x1),2) - np.power(x2,2))),2) - 0.5 
    term2 = (1 + 0.001 * (np.power(x1,2) + np.power(x2,2))) **2
    s = 0.5 + (term1 / term2)
    return s

'''Schaffer No.4 '''    
def F7(x):
    x = np.asarray_chkfinite(x)
    x1 = x[0]
    x2 = x[1]
    term1 = np.power(np.cos(np.sin(np.power(np.abs(x1),2) - np.power(x2,2))),2) - 0.5 
    term2 = (1 + 0.001 * (np.power(x1,2) + np.power(x2,2))) **2
    s = 0.5 + term1 / term2;
    return s 

'''Zakhrov'''
def F8(x):
    x = np.asarray_chkfinite(x)
    n = len(x);
    term1 = 0;
    term2 = 0;    
    for i in range(0,n):
        term1 = term1 + (np.power(x[i],2))
        term2 = term2 + (0.5 * i * x[i])
    s = term1 + (np.power(term2,2)) + (np.power(term2,4))
    return s

'''Quartic'''    
def F9(x):
    x = np.asarray_chkfinite(x)
    w=[i for i in range(len(x))]
    np.add(w,1)
    s = np.sum(np.multiply(w,np.power(x,4)) + np.random.uniform(0,1))
    return s

'''Schwefel 2.21 -To test'''
def F10(x):
    x = np.asarray_chkfinite(x)
    w=len(x)
    max=0.0
    for i in range(0,w):
        if abs(x[i])>max:
            max= abs(x[i])
            return max

'''Schwefel 2.22  -To test'''
def F11(x):
    x = np.asarray_chkfinite(x)
    term1 = 0.0
    term2 = 1.0
    w=len(x)
    for i in range(w):
        term1 += abs(x[i])
        term2 *= abs(x[i])
    s=term1 + term2
    return s

'''sphere'''
def F12( x ):
    s = np.asarray_chkfinite(x)
    return np.sum( s**2.0 )      

'''step2'''
def F13( x ):
    x=np.asarray_chkfinite(x)
    s=np.sum(np.floor((x+.5))**2)
    return s

'''stepint'''
def F14(x):
    x=np.asarray_chkfinite(x)
    s = np.sum(np.ceil(x)) + 25
    return s

'''sumsquares'''
def F15(x):
    x=np.asarray_chkfinite(x)
    w=len(x)
    p=0
    for i in range(0,w):
        p=p+ np.multiply(i,np.power(x[i],2))
    s=p
    return s

'''ackley'''
def F16(x):
    term1 = 0.0
    term2 = 0.0
    x=np.asarray_chkfinite(x)
    for c in x:
        term1 += np.power(c,2.0)
        term2 += np.cos(2.0*np.pi*c)
    n = float(len(x))
    s= -20.0*np.exp(-0.2*np.sqrt(term1/n)) - np.exp(term2/n) + 20 + np.e
    return s

'''Bohachevsky no.2'''
def F17(x):
    x1 = x[0]
    x2 = x[1]
    s= (np.power(x1,2)) + (2*np.power(x2,2)) - (0.2* np.cos(0.3*np.pi*x1))*np.cos(4*np.pi*x2) + 0.3
    #for x1, x2 in zip(x[:-1], x[1:])),
    return s

'''Bohachevsky no.3'''
def F18(x):
    x1 = x[0]
    x2 = x[1]
    s=np.power(x1,2) + 2*np.power(x2,2) - 0.3*np.cos(3*np.pi*x1+ 4*np.pi*x2)
    return s        

'''Crossintray'''
def F19(x):
    x1 = x[0]
    x2 = x[1]
#    x1=float(x1)
#    x2=float(x2)
    a=np.sqrt(np.power(x1,2)*np.power(x2,2))
    expo = np.abs(100 - (a/np.pi))
    inside = np.fabs(np.sin(x1) * np.sin(x2) * expo) + 1
    s = (-0.0001) * np.power(inside, 0.1)
    return s

'''Griewank'''    
def F20(x):
    fr=4000
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    temp_sum = sum( np.power(x,2))
    p = np.prod(np.cos( x / np.sqrt(j) ))
    s=temp_sum/fr - p + 1
    return s 

'''GoldStein-Price'''
def F21(x):
    x1 = x[0]
    x2 = x[1]
    s = (1+ (x1+x2+1)**2.0* (19- 14*x1+ 3 *x1** 2.0- 14*x2+ 6*x1*x2 + 3 *x2**2.0)) * (30+ (2*x1-3*x2)**2.0* (18-32*x1+12*x1**2.0+ 48*x2- 36 * x1*x2+ 27*x2**2.0))
    return s

'''Hartman 3'''
def F22(x):
    x = np.asarray_chkfinite(x)
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = np.array([[3.0, 10.0, 30.0],
                       [0.1, 10.0, 35.0],
                       [3.0, 10.0, 30.0],
                       [0.1, 10.0, 35.0]])
    P = 0.0001 * np.array([[3689, 1170, 2673],
                                [4699, 4387, 7470],
                                [1090, 8732, 5547],
                                [381, 5743, 8828]])
    extern_sum = 0
    for i in range(4):
        intern_sum=0
        for j in range(3):
            intern_sum = intern_sum+ A[i, j]*np.power((x[j]-P[i, j]),2)
        extern_sum = extern_sum + alpha[i] * np.exp(-intern_sum)
        s=-extern_sum
    return s 

'''Hartman 6'''
def F23(x):
    alpha = [1.00, 1.20, 3.00, 3.20]
    A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                      [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                      [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                      [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
    P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])
    extern_sum = 0
    for i in range(4):
        intern_sum = 0
        for j in range(6):
            intern_sum = intern_sum + A[i, j] * (x[j] - P[i, j]) ** 2
            extern_sum = extern_sum + alpha[i] * np.exp(-intern_sum)
            s=-extern_sum
            return s

'''Penalized no 1 - BUG TO BE FIXED AND TESTED'''
def F24(x):
    x = np.asarray_chkfinite(x)
    w = len(x)
#    term1=np.pi/w
#    term2 = (10*((np.sin(np.pi*(1+(x[0]+1)/4)))**2))
#    term3= sum((((x[0:w-1]+1)/4)**2))
#    term4 = (1+10*((np.sin(np.pi*(1+(x[1:w]+1)/4))))**2)
#    term5 = ((x[w-1]+1)/4)**2
#    term6 = sum(fun_penalized(x,10,100,4))
#    s= (term1)*(term2+((term3)*(term4))+(term5))+ (term6)
#    (term1)*(term2)+term3*(term4)+(term5)+term6
    #y = (pi/dim)*(10*((sin(pi*(1+(x(1)+1)/4)))^2)+sum((((x(1:dim-1)+1)./4).^2).*...
     #   (1+10.*((sin(pi.*(1+(x(2:dim)+1)./4)))).^2))+((x(dim)+1)/4)^2)+sum(Ufun(x,5,100,4));
                  
    s = (np.pi/w)*(10*((np.sin(np.pi*(1+(x[0]+1)/4)))**2)+np.sum((((x[0:w-1]+1)/4)**2)*
                   (1+10*((np.sin(np.pi*(1+(x[1:w]+1)/4))))**2))+((x[w-1]+1)/4)*2)+sum(fun_penalized(x,5,100,4))
    return s

'''Penalized no 2'''
def F25(x):
    x = np.asarray_chkfinite(x)
    w = len(x)
    #a=10;k=100;m=4;
    #ufun = lambda x: k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<(-a))
    
    s =0.1*((np.sin(3*np.pi*x[0]))**2+ sum((x[0:w-1]-1)**2 * (1+(np.sin(3*np.pi*x[1:w]))**2))+
            ((x[w-1]-1)**2)*(1+(np.sin(2*np.pi*x[w-1]))**2))+sum(fun_penalized(x,5,100,4))
    #s =0.1*((np.sin(3*np.pi*x[0]))**2+ sum((x[0:w-1]-1)**2 * (1+(np.sin(3*np.pi*x[1:w]))**2))+
     #       ((x[w-1]-1)**2)*(1+(np.sin(2*np.pi*x[w-1]))**2))+ufun(x)
    return s

def fun_penalized(x,a,k,m):
    s=k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<(-a))
    return s

'''Perm''' 
def F26(x):
    x = np.asarray_chkfinite(x)
    b = 0.5;d = len(x);s = 0;
    for ii in range(d):
        i = 0
        for jj in range(d):
            xj = x[jj]
            i = i + ((jj+1)+b) * xj**(ii+1) - (1/(jj+1))**(ii+1)
        s=s+(i)**2
    return s

'''Powersum'''
def F27(x):  
    x = np.asarray_chkfinite(x)
    n = len(x)
    s = 0
    b=[8,18,44,114]
    for k in range( 1, n+1 ):
        bk = b[ min( k - 1, len(b) - 1 )]  # ?
        s += (sum( x**k ) - bk) **2  # dim 10 huge, 100 overflows
    return s

'''Shubert'''
def F28(x):
    x = np.asarray_chkfinite(x)
    x1 = x[0]
    x2 = x[1]
    s1 = 0
    s2 = 0
    for i in range(5):
        n1= (i+1)*np.cos((i+2)*x1+(i+1))
        n2= (i+1)*np.cos((i+2)*x2+(i+1))
        s1=s1+n1
        s2=s2+n2
    s = s1 * s2
    return s

'''Alpine No.1'''
def F29(x):
    x = np.asarray_chkfinite(x)
    s = sum(abs(x*np.sin(x)+0.1*x))
    return s

'''BohachevskyNo.1'''
def F30(x):
    x = np.asarray_chkfinite(x)
    x1=x[0]
    x2=x[1]
    s= (x1**2) + (2*x2**2) - (0.3 * np.cos(3*np.pi*x1)) - (0.4* np.cos(4*np.pi*x2)) + 0.7
    return s

'''Booth'''
def F31(x):
     x = np.asarray_chkfinite(x)
     s=(x[0]**2*x[1]-7)**2+(2*x[0]+x[1]-5)**2
     return s

'''Branin'''
def F32(x):
    a = 1; b =  5.1/(4*np.pi**2); c =  5/np.pi; r = 6; p =  10; t =  1/(8*np.pi);
    x = np.asarray_chkfinite(x)
    x1=x[0]
    x2=x[1]
    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = p*(1-t)*np.cos(x1)
    s = term1 + term2 + p
    return s

'''Michalewics 2'''
def F33(x):
    n = 2; m = 10;s = 0;
    x = np.asarray_chkfinite(x)
    for i in range(n):
        s= s+ np.sin(x[i])*(np.sin((i+1)* x[i]** 2/np.pi))**(2*m)
    return -s

'''Michalewics 5'''
def F34(x):
    n = 5; m = 10;s = 0;
    x = np.asarray_chkfinite(x)
    for i in range(n):
        s= s+ np.sin(x[i])*(np.sin((i+1)* x[i]** 2/np.pi))**(2*m)
    return -s

'''Michalewics 5'''
def F35(x):
    n = 10; m = 10;s = 0;
    x = np.asarray_chkfinite(x)
    for i in range(n):
        s= s+ np.sin(x[i])*(np.sin((i+1)* x[i]** 2/np.pi))**(2*m)
    return -s

'''Rastrigin'''
def F36(x):  
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 10*n + sum( x**2 - 10 * np.cos( 2 * np.pi * x ))


'''INPUTS - 
    clasific_regr= 1 OR 0. 1= Classification problem, 0 = Regression Problem;
    kernel - linear, polynomial or rbf kernel ; 
    X_tr,X_te,y_tr,y_te = Train and Test data ;
    c_pos,gamma_pos = C and Gamma value array
   OUTPUT - Fitness 
'''

'''INPUTS - 
    clasific_regr= 1 OR 0. 1= Classification problem, 0 = Regression Problem;
    kernel - linear, polynomial or rbf kernel ; 
    X_tr,X_te,y_tr,y_te = Train and Test data ;
    c_pos,gamma_pos = C and Gamma value array
   OUTPUT - Fitness 
'''
'''def F36(clasific_regr,svm_kernel,X_tr,X_te,y_tr,y_te ,c_pos,gamma_pos):
    clasific_regr=1
    svm_kernel='rbf'
    if clasific_regr == 1:
        rbf_regressor = svm.SVR(kernel = svm_kernel, C = c_pos, gamma = gamma_pos).fit(X_tr, y_tr)
        cv_scores = cross_val_score(rbf_regressor,X_te,y_te,cv =3,scoring = 'neg_mean_squared_error') # Taking negated value of MSE
        #To minimize the error rate
        scores = cv_scores.mean()
        fitness = (1 - scores)*100
    else:
        rbf_svm = svm.SVC(kernel = svm_kernel, C = c_pos, gamma = gamma_pos).fit(X_tr, y_tr)  #svm
        cv_scores = cross_val_score(rbf_svm,X_te,y_te,cv =3,scoring = 'accuracy')
        #To minimize the error rate
        scores = cv_scores.mean()
        fitness = (1 - scores)*100
    return fitness
'''

def ProblemsDetails(a):
    param = {  "F1":["F1",-4.5,4.5,2],
         "F2":["F2",-10,10,2], 
         "F3":["F3",-10,10,2],
         "F4":["F4",-4,5,24],
         "F5":["F5",-100,100,2],#To be changed - All below values
         "F6":["F6",-100,100,2],
         "F7":["F7",-100,100,2],
         "F8":["F8",-5,10,30],
         "F9":["F9",-1.28,1.28,30],
         "F10":["F10",-100,100,30],
         "F11":["F11",-10,10,30],
         "F12":["F12",-100,100,30],
         "F13":["F13",-100,100,30],
         "F14":["F14",-5.12,5.12,5],
         "F15":["F15",-10,10,30],
         "F16":["F16",-32,32,30],
         "F17":["F17",-10,10,2],
         "F18":["F18",-100,100,2],
         "F19":["F19",-10,10,2],
         "F20":["F20",-600,600,30],
         "F21":["F21",-2,2,2],
         "F22":["F22",0,1,3],
         "F23":["F23",0,1,6],
         "F24":["F24",-50,50,30],
         "F25":["F25",-50,50,30],
         "F26":["F26",-4,4,4],
         "F27":["F27",0,4,4],
         "F28":["F28",-10,10,2],
         "F29":["F29",-10,10,30],
         "F30":["F30",-100,100,2],
         "F31":["F31",-10,10,2],
         "F32":["F32",-5,5,2],
         "F33":["F33",0,np.pi,2],
         "F34":["F34",0,np.pi,5],
         "F35":["F35",0,np.pi,10],
         "F36":["F36",-5.12,5.12,30]
}
    return param.get(a, "nothing")
