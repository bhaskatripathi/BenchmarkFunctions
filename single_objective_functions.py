# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 00:45:04 2020

@author: Bhaskar Tripathi

Test functions in this file are mostly derived from https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import math
import numpy as np
#from dalo_optimizer import DALO
import matplotlib.pyplot as plt
import os


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
    """
    Schaffer's First Function
    f(x) = 0.5 + (sin^2(x1^2 + x2^2)^2 - 0.5) / (1 + 0.001(x1^2 + x2^2)^2)
    Domain: -100 <= x_i <= 100
    Global minimum at (0,0) with f(x*) = 0
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    
    numerator = np.sin((x1**2 + x2**2))**2 - 0.5  # Note: sum of squares
    denominator = (1 + 0.001 * (x1**2 + x2**2)**2)
    return 0.5 + numerator/denominator

'''Schaffer No. 3'''    
def F6(x):
    """
    Schaffer's Second Function
    f(x) = 0.5 + (sin^2(x1^2 - x2^2)^2 - 0.5) / (1 + 0.001(x1^2 + x2^2)^2)
    Domain: -100 <= x_i <= 100
    Global minimum at (0,0) with f(x*) = 0
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    
    numerator = np.sin((x1**2 - x2**2))**2 - 0.5  # Note: difference of squares
    denominator = (1 + 0.001 * (x1**2 + x2**2)**2)
    return 0.5 + numerator/denominator

'''Schaffer No.4 '''    
def F7(x):
    """
    Schaffer's Third Function
    f(x) = 0.5 + (sin^2(cos|x1^2 - x2^2|) - 0.5) / (1 + 0.001(x1^2 + x2^2)^2)
    Domain: -100 <= x_i <= 100
    Global minimum at (0,1.253115) with f(x*) = 0.00156685
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    
    numerator = np.sin(np.cos(np.abs(x1**2 - x2**2)))**2 - 0.5  # Note: cos of absolute difference
    denominator = (1 + 0.001 * (x1**2 + x2**2)**2)
    return 0.5 + numerator/denominator

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
def F100(x):
    x = np.asarray_chkfinite(x)
    w=len(x)
    max=0.0
    for i in range(0,w):
        if abs(x[i])>max:
            max= abs(x[i])
            return max

def F10(x):
    x = np.asarray_chkfinite(x)
    return np.max(np.abs(x))

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

'''Weierstrass Function'''
def F37(x):
    """
    Weierstrass Function
    f(x) = sum[sum[a^k cos(2π b^k(x_i + 0.5))] - D sum[a^k cos(2π b^k)]]
    Domain: -0.5 <= x_i <= 0.5
    Kmax should be 100
    Global minimum at (0,...,0) with f* = 0
    """
    x = np.asarray_chkfinite(x)
    a = 0.5
    b = 3
    Kmax = 100  # As specified in the Nature inpired computing book by Sin she yang
    n = len(x)
    
    # First sum term
    sum1 = 0
    for i in range(n):
        for k in range(Kmax + 1):
            sum1 += a**k * np.cos(2*np.pi*b**k*(x[i] + 0.5))
    
    # Second sum term
    sum2 = 0
    for k in range(Kmax + 1):
        sum2 += a**k * np.cos(2*np.pi*b**k)
    
    return sum1 - n*sum2

'''Rotated Ackley (F39)'''
def F38(x):
    x = np.asarray_chkfinite(x)
    # Define a rotation matrix (example: rotate by 45 degrees in the first two dimensions)
    # For higher dimensions, consider a more complex rotation matrix.
    theta = np.pi/4
    R = np.eye(len(x))
    if len(x) > 1:
        R[0,0] = np.cos(theta); R[0,1] = -np.sin(theta)
        R[1,0] = np.sin(theta); R[1,1] = np.cos(theta)
    x_rot = R.dot(x)
    
    # Ackley calculation on rotated coordinates
    n = float(len(x_rot))
    term1 = np.sum(x_rot**2.0)/n
    term2 = np.sum(np.cos(2.0*np.pi*x_rot))/n
    return -20.0*np.exp(-0.2*np.sqrt(term1)) - np.exp(term2) + 20 + np.e

'''Rotated Rastrigin (F40):'''
def F39(x):
    x = np.asarray_chkfinite(x)
    # Define a rotation matrix (same logic as above)
    theta = np.pi/4
    R = np.eye(len(x))
    if len(x) > 1:
        R[0,0] = np.cos(theta); R[0,1] = -np.sin(theta)
        R[1,0] = np.sin(theta); R[1,1] = np.cos(theta)
    x_rot = R.dot(x)

    n = len(x_rot)
    return 10*n + np.sum(x_rot**2 - 10 * np.cos(2*np.pi*x_rot))

def F40(x):  # Katsuura
    x = np.asarray_chkfinite(x)
    n = len(x)
    product = 1.0
    for i in range(n):
        term = 1.0
        for j in range(1, 33):
            term += abs(2**j * x[i] - round(2**j * x[i])) / (2**j)
        product *= (1 + (i+1)*term)
    return (10 / n**2) * product - (10 / n**2)

def F41(x):  # Whitley Function
    """
    Whitley Function
    Combines a very steep overall slope with a highly multimodal area
    around the global minimum located at x_i = 1 where i = 1,...,D
    Domain: -10 <= x_i <= 10
    Global minimum at (1,1,...,1)
    """
    x = np.asarray_chkfinite(x)
    n = len(x)
    s = 0.0
    for i in range(n):
        for j in range(n):
            term1 = (100*(x[i]**2 - x[j])**2 + (1 - x[j])**2)
            s += term1/4000 - np.cos(term1) + 1
    return s

def F42(x):  # Salomon
    x = np.asarray_chkfinite(x)
    sq_sum = np.sum(x**2)
    root_val = np.sqrt(sq_sum)
    return 1 - np.cos(2 * np.pi * root_val) + 0.1 * root_val

def F43(x):  # Langermann
    # Constants as per standard definition:
    m = 5
    A = np.array([3, 5, 2, 1, 7])
    B = np.array([5, 2, 1, 4, 9])
    C = 1/np.array([1,2,5,2,3])**2  # Or other scaling factors depending on definition
    x = np.asarray_chkfinite(x)
    s = 0.0
    for i in range(m):
        s += (C[i]) * np.exp(-1/(np.pi)*( (x[0]-A[i])**2+(x[1]-B[i])**2 )) * \
             np.cos(np.pi*((x[0]-A[i])**2+(x[1]-B[i])**2))
    return -s

def F44(x):  # Kowalik
    x = np.asarray_chkfinite(x)
    # Known constants for Kowalik function
    y = np.array([0.1957,0.1947,0.1735,0.1600,0.0844,0.0627,0.0456,0.0342,0.0235,0.0246])
    u = np.array([0.25,0.5,1.0,2.0,4.0,6.0,8.0,10.0,12.0,14.0])
    numerator = (y - (x[0]*(u**2 + u*x[1])) / (u**2 + u*x[2] + x[3]))**2
    return np.sum(numerator)

def F45(x):  # Bartels Conn
    """
    Bartels Conn Function
    f(x) = |x1^2 + x2^2 + x1*x2| + |sin(x1)| + |cos(x2)|
    Domain: -500 <= x_i <= 500
    Global minimum at (0,0) with f(0,0) = 1
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    return abs(x1**2 + x2**2 + x1*x2) + abs(np.sin(x1)) + abs(np.cos(x2))


def F46(x):  # Bird
    """
    Bird Function
    f(x) = sin(x1)*exp((1 - cos(x2))^2) + cos(x2)*exp((1 - sin(x1))^2) + (x1 - x2)^2
    Domain: -2π <= x_i <= 2π
    Has two global minima at (4.70104, 3.15294) and (-1.58214, -3.13024), f* ~ -106.764537
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    term1 = np.sin(x1) * np.exp((1 - np.cos(x2))**2)
    term2 = np.cos(x2) * np.exp((1 - np.sin(x1))**2)
    term3 = (x1 - x2)**2
    return term1 + term2 + term3


def F47(x):  # Box-Betts
    """
    Box–Betts Quadratic Sum Function
    f(x) = sum_{i=0 to 9} [exp(-0.1*(i+1)*x1) - exp(-0.1*(i+1)*x2)
                           - (exp(-0.1*(i+1)) - exp(-(i+1)*x3))]^2
    Typically: 0.9 <= x1 <= 1.2, 9 <= x2 <= 11.2, 0.9 <= x3 <= 1.2
    Global minimum at (1, 10, 1) with f* = 0
    """
    x = np.asarray_chkfinite(x)
    x1, x2, x3 = x
    s = 0.0
    for i in range(10):
        a = np.exp(-0.1*(i+1)*x1)
        b = np.exp(-0.1*(i+1)*x2)
        c = np.exp(-0.1*(i+1)) - np.exp(-(i+1)*x3)
        g = a - b - c
        s += g*g
    return s


def F48(x):  # Bukin
    """
    Bukin Function (one common variant)
    f(x) = 100 [x2 - 0.01*x1^2 + 1] + 0.01(x1 + 10)^2
    Domain constraints:
    - x1 should be in [-15, -5]
    - x2 should be in [-3, 3]
    Global min at (-10, 0) with f* = 0
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    
    # Add penalty for solutions outside the intended domain
    penalty = 0
    if x1 < -15 or x1 > -5:
        penalty += 1e6
    if x2 < -3 or x2 > 3:
        penalty += 1e6
        
    return 100*(x2 - 0.01*x1**2 + 1) + 0.01*(x1 + 10)**2 + penalty


def F49(x):  # Camel (Three-Hump Camel Function)
    """
    Three-Hump Camel
    f(x) = 2*x1^2 - 1.05*x1^4 + (x1^6)/6 + x1*x2 + x2^2
    Domain: -5 <= x_i <= 5
    Global minimum at (0,0) with f* = 0
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    return 2*x1**2 - 1.05*x1**4 + (x1**6)/6 + x1*x2 + x2**2


def F50(x):  # Camel (Six-Hump Camel Function)
    """
    Six-Hump Camel
    f(x) = (4 - 2.1*x1^2 + (x1^4)/3) * x1^2 + x1*x2 + (4*x2^2 - 4)*x2^2
    Domain: -5 <= x_i <= 5
    Global minima at (+-0.0898, 0.7126) with f* ~ -1.0316
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    return (4 - 2.1*x1**2 + (x1**4)/3)*x1**2 + x1*x2 + (4*x2**2 - 4)*x2**2


def F51(x):  # Chichinadze
    """
    Chichinadze Function
    f(x) = x1^2 - 12*x1 + 11 + 10 cos(π x1/2) + 8 sin(5π x1/2)
           - sqrt(1/5)*exp(-0.5*(x2 - 0.5)^2)
    Domain: -30 <= x1 <= 30, often x2 in same range
    Global min at (x1, x2) = (5.90133, 0.5) with f* ~ -43.3159
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    return (x1**2 - 12*x1 + 11
            + 10*np.cos((np.pi*x1)/2)
            + 8*np.sin((5*np.pi*x1)/2)
            - np.sqrt(1/5)*np.exp(-0.5*(x2 - 0.5)**2))


def F52(x):  # Cosine Mixture
    """
    Cosine Mixture Function
    f(x) = -0.1 sum(cos(5πx_i)) - sum(x_i^2)
    Domain: -1 <= x_i <= 1
    Global minimum at (0,...,0) with f* = 0.2 for D=2
    """
    x = np.asarray_chkfinite(x)
    return -0.1 * np.sum(np.cos(5 * np.pi * x)) - np.sum(x**2)

def F53(x):  # Csendes
    """
    Csendes Function
    f(x) = sum(x_i^6 * (2 + sin(1/x_i)))
    Domain: -1 <= x_i <= 1
    Global minimum at (0,...,0) with f* = 0
    """
    x = np.asarray_chkfinite(x)
    # Avoid division by zero by adding small epsilon where x is zero
    x = np.where(x == 0, 1e-10, x)
    return np.sum(x**6 * (2 + np.sin(1/x)))

def F54(x):  # Cube
    """
    Cube Function
    f(x) = 100(x2 - x1^3)^2 + (1 - x1)^2
    Domain: -10 <= x_i <= 10
    Global minimum at (-1,1) with f* = 0
    """
    x = np.asarray_chkfinite(x)
    return 100 * (x[1] - x[0]**3)**2 + (1 - x[0])**2

def F55(x):  # Watson
    """
    Watson Function
    Complex sum function with coefficients a_i = i/29
    Domain: -10 <= x_i <= 10
    Global minimum at (-0.0158,1.012,-0.2329,1.260,-1.513,0.9928)
    """
    x = np.asarray_chkfinite(x)
    n = len(x)
    sum = 0
    
    for i in range(30):  # i from 0 to 29
        sum1 = 0
        sum2 = 0
        for j in range(n):
            sum1 += (j) * (i/29.0)**(j) * x[j]  # Note: j starts from 0
        for j in range(n):
            sum2 += (i/29.0)**(j) * x[j]
        sum += (sum1 - sum2**2 - 1)**2
    
    sum += x[0]**2
    return sum

def F56(x):  # Xin-She Yang First Function
    """
    Xin-She Yang First Function
    f(x) = sum(epsilon_i * |x_i|^i) where epsilon_i is random in [0,1]
    Domain: -5 <= x_i <= 5
    Global minimum at (0,...,0) with f* = 0
    """
    x = np.asarray_chkfinite(x)
    n = len(x)
    # Generate random coefficients (fixed seed for reproducibility)
    np.random.seed(42)  # Fixed seed for consistent results
    epsilon = np.random.uniform(0, 1, n)
    return np.sum(epsilon * np.abs(x) ** np.arange(1, n+1))

def F57(x):  # Xin-She Yang Second Function
    """
    Xin-She Yang Second Function
    f(x) = (sum(|x_i|)) * exp(-sum(sin(x_i^2)))
    Domain: -2π <= x_i <= 2π
    Global minimum at (0,...,0) with f* = 0
    """
    x = np.asarray_chkfinite(x)
    return np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2)))

def F58(x):  # Xin-She Yang Third Function
    """
    Xin-She Yang Third Function
    f(x) = exp(-sum((x_i/β)^(2m))) - 2exp(-sum(x_i^2)) * prod(cos^2(x_i))
    Domain: -20 <= x_i <= 20
    Global minimum at (0,...,0) with f* = -1
    Parameters: m = 5, β = 15
    """
    x = np.asarray_chkfinite(x)
    m = 5
    beta = 15
    term1 = np.exp(-np.sum((x/beta)**(2*m)))
    term2 = 2 * np.exp(-np.sum(x**2)) * np.prod(np.cos(x)**2)
    return term1 - term2

def F59(x):  # Xin-She Yang Fourth Function
    """
    Xin-She Yang Fourth Function
    f(x) = (sum(sin^2(x_i) - exp(-sum(x_i^2))) * exp(-sum(sin^2(sqrt(|x_i|))))
    Domain: -10 <= x_i <= 10
    Global minimum at (0,...,0) with f* = -1
    """
    x = np.asarray_chkfinite(x)
    term1 = np.sum(np.sin(x)**2) - np.exp(-np.sum(x**2))
    term2 = np.exp(-np.sum(np.sin(np.sqrt(np.abs(x)))**2))
    return term1 * term2

def F60(x):  # Xin-She Yang Fifth Function
    """
    Xin-She Yang Fifth Function
    f(x) = |1 - exp(-sum(epsilon_i * x_i^2))| + sum(epsilon_i * sin^2(2π*x_i))
    Domain: -10π <= x_i <= 10π
    Global minimum at (0,...,0) with f* = 0
    """
    x = np.asarray_chkfinite(x)
    # Generate random coefficients (fixed seed for reproducibility)
    np.random.seed(42)  # Fixed seed for consistent results
    epsilon = np.random.uniform(0, 1, len(x))
    
    term1 = np.abs(1 - np.exp(-np.sum(epsilon * x**2)))
    term2 = np.sum(epsilon * np.sin(2 * np.pi * x)**2)
    return term1 + term2

def F61(x):  # Wood Function
    """Similar to Powell, has interdependent variables
    Domain: -4 <= x_i <= 4
    Global minimum at (1,1,1,1) with f(x)=0"""
    x = np.asarray_chkfinite(x)
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return (100*(x2-x1**2)**2 + (1-x1)**2 + 
           90*(x4-x3**2)**2 + (1-x3)**2 +
           10.1*((x2-1)**2 + (x4-1)**2) +
           19.8*(x2-1)*(x4-1))

def F62(x):  # Vincent Function
    """Highly oscillatory like Schaffer functions
    Domain: 0.25 <= x_i <= 10
    Multiple global minima"""
    x = np.asarray_chkfinite(x)
    return -np.sum(np.sin(10*np.log(x)))/len(x)

def F63(x):  # Egg Holder Function
    """Highly oscillatory with multiple local minima
    Domain: -512 <= x_i <= 512"""
    x = np.asarray_chkfinite(x)
    return -np.sum((x[1:] + 47)*np.sin(np.sqrt(np.abs(x[1:] + x[:-1]/2 + 47))) +
                   x[:-1]*np.sin(np.sqrt(np.abs(x[:-1] - (x[1:] + 47)))))

def F64(x):  # Noisy Quartic with Gaussian
    """Similar to Quartic but with Gaussian noise
    Domain: -1.28 <= x_i <= 1.28"""
    x = np.asarray_chkfinite(x)
    n = len(x)
    return np.sum(np.arange(1,n+1)*x**4) + np.random.normal(0, 0.1)

def F65(x):  # Modified Schwefel
    """Similar characteristics to Schwefel functions
    Domain: -100 <= x_i <= 100"""
    x = np.asarray_chkfinite(x)
    return 418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))

def F66(x):  # Alpine Modified
    """Combines absolute values like Schwefel
    Domain: -10 <= x_i <= 10"""
    x = np.asarray_chkfinite(x)
    return np.sum(np.abs(x*np.sin(x) + 0.1*x))

def F67(x):  # Tripod
    """
    Tripod Function
    f(x) = p(x2)(1 + p(x1)) + |x1 + 50p(x2)(1 - 2p(x1))| + |x2 + 50(1 - 2p(x2))|
    where p(x) = 1 for x ≥ 0, and 0 otherwise
    Domain: -100 <= x_i <= 100
    Global minimum at (0,-50) with f(x*) = 0
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    
    def p(x):
        return 1.0 if x >= 0 else 0.0
    
    term1 = p(x2) * (1 + p(x1))
    term2 = abs(x1 + 50 * p(x2) * (1 - 2 * p(x1)))
    term3 = abs(x2 + 50 * (1 - 2 * p(x2)))
    
    return term1 + term2 + term3

def F68(x):  # Ursem
    """
    Ursem Function
    f(x) = -sin(2x1 - 0.5π) - 3cos(x2) - 0.5x1
    Domain: -2.5 <= x1 <= 3 and -2 <= x2 <= 2
    Has a single global optimum
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    
    return -np.sin(2*x1 - 0.5*np.pi) - 3*np.cos(x2) - 0.5*x1

def F69(x):  # Ursem Waves
    """
    Ursem Waves Function
    f(x) = -0.9x1^2 + (x2^2 - 4.5x2^2)x1x2 + 4.7cos(3x1 - x2^2(2 + x1))sin(2.5πx1)
    Domain: -0.9 <= x1 <= 1.2 and -1.2 <= x2 <= 1.2
    Has a single global minimum and nine irregularly spaced local minima
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    
    term1 = -0.9 * x1**2
    term2 = (x2**2 - 4.5*x2**2) * x1 * x2
    term3 = 4.7 * np.cos(3*x1 - x2**2 * (2 + x1)) * np.sin(2.5*np.pi*x1)
    
    return term1 + term2 + term3

def F70(x):  # TestTube Holder
    """
    TestTube Holder Function
    f(x) = -4[(sin(x1)cos(x2))exp(cos[(x1^2 + x2^2)/200])]
    Domain: -10 <= x_i <= 10
    Global minima at x* = (±π/2,0) with f(x*) = -10.87230
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    
    return -4 * (np.sin(x1) * np.cos(x2) * np.exp(np.cos((x1**2 + x2**2)/200)))

def F71(x):  # Trecanni
    """
    Trecanni Function
    f(x) = x1^4 - 4x1^3 + 4x1 + x2^2
    Domain: -5 <= x_i <= 5
    Global minima at x* = (0,0) and (-2,0) with f(x*) = 0
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    
    return x1**4 - 4*x1**3 + 4*x1 + x2**2

def F72(x):  # Trid
    """
    Trid Function
    f(x) = sum(i=1 to D)[(xi - 1)^2] - sum(i=1 to D)[xi*x(i-1)]
    Domain: -6^2 <= x_i <= 6^2
    Global minimum at f(x*) = -50
    """
    x = np.asarray_chkfinite(x)
    D = len(x)
    
    sum1 = np.sum((x - 1)**2)
    sum2 = np.sum(x[1:] * x[:-1])
    
    return sum1 - sum2

def F73(x):  # Trefethen
    """
    Trefethen Function
    f(x) = exp(sin(50x1)) + sin(60exp(x2)) + sin(70sin(x1)) + sin(sin(80x2))
          - sin(10(x1 + x2)) + 1/4(x1^2 + x2^2)
    Domain: -10 <= x_i <= 10
    Global minimum at x* = (-0.024403,0.210612) with f(x*) = -3.30686865
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    
    term1 = np.exp(np.sin(50*x1))
    term2 = np.sin(60*np.exp(x2))
    term3 = np.sin(70*np.sin(x1))
    term4 = np.sin(np.sin(80*x2))
    term5 = -np.sin(10*(x1 + x2))
    term6 = 0.25*(x1**2 + x2**2)
    
    return term1 + term2 + term3 + term4 + term5 + term6

def F74(x):  # Styblinski-Tang Function
    """
    Styblinski-Tang Function
    f(x) = 0.5 * sum(x^4 - 16x^2 + 5x)
    Domain: -5 <= x_i <= 5
    Global minimum at (-2.903534, -2.903534) with f* = -78.332
    """
    x = np.asarray_chkfinite(x)
    return 0.5 * np.sum(x**4 - 16*x**2 + 5*x)

def F75(x):  # First Holder Table Function
    """
    First Holder Table Function
    f(x) = -|cos(x1)cos(x2)exp(|1-(sqrt(x1^2+x2^2))/π|)|
    Domain: -10 <= x_i <= 10
    Global minima at (±9.646168, ±9.646168) with f* = -26.920336
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    term = np.abs(1 - np.sqrt(x1**2 + x2**2)/np.pi)
    return -np.abs(np.cos(x1) * np.cos(x2) * np.exp(term))

def F76(x):  # Second Holder Table Function
    """
    Second Holder Table Function
    f(x) = -|sin(x1)cos(x2)exp(|1-(sqrt(x1^2+x2^2))/π|)|
    Domain: -10 <= x_i <= 10
    Global minima at (±8.05502,±9.66459) with f* = -19.2085
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    term = np.abs(1 - np.sqrt(x1**2 + x2**2)/np.pi)
    return -np.abs(np.sin(x1) * np.cos(x2) * np.exp(term))

def F77(x):  # Carrom Table Function
    """
    Carrom Table Function
    f(x) = -((cos(x1)cos(x2)exp(|1-sqrt(x1^2+x2^2)/π|))^2)/30
    Domain: -10 <= x_i <= 10
    Global minima at (±9.646157,±9.646157) with f* = -24.1568155
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    term = np.abs(1 - np.sqrt(x1**2 + x2**2)/np.pi)
    inner = np.cos(x1) * np.cos(x2) * np.exp(term)
    return -(inner**2)/30

def F78(x):  # Shekel Function
    """
    Shekel Function (also known as Shekel's foxholes)
    f(x) = sum(1/(c_i + sum((x_j - a_ij)^2)))^(-1)
    Domain: Usually [0, 10]^n
    The function has m maxima where m is configurable
    Common configuration: n=2 dimensions with m=10 maxima
    """
    x = np.asarray_chkfinite(x)
    
    # Define parameters for 10 maxima in 2D (can be modified for different m and n)
    m = 10  # number of maxima
    n = len(x)
    
    # Define c parameters
    c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    
    # Define a matrix of parameters (commonly used values for 2D)
    a = np.array([
        [4, 4],
        [1, 1],
        [8, 8],
        [6, 6],
        [3, 7],
        [2, 9],
        [5, 5],
        [8, 1],
        [6, 2],
        [7, 3.6]
    ])
    
    result = 0
    for i in range(m):
        term = c[i]
        for j in range(n):
            term += (x[j] - a[i,j])**2
        result += 1.0/term
    
    return -result  # Negative because we're minimizing

def F79(x):  # Trigonometric Function
    """
    Trigonometric Function
    f(x) = sum[D - sum(cos(x_j) + i(1-cos(x_i) - sin(x_i)))]^2
    Domain: 0 <= x_i <= pi
    Global minimum at x* = (0,...,0) with f(x*) = 0
    """
    x = np.asarray_chkfinite(x)
    D = len(x)
    
    outer_sum = 0
    for i in range(D):
        inner_sum = np.sum(np.cos(x))  # sum of cos(x_j) for all j
        term = D - inner_sum + i * (1 - np.cos(x[i]) - np.sin(x[i]))
        outer_sum += term**2
    
    return outer_sum

def F80(x):  # Rosenbrock Function
    """
    Rosenbrock Function (also known as Banana function)
    f(x) = sum[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2] for i = 1 to n-1
    Domain: Usually [-2.048, 2.048]
    Global minimum:
    n = 2  -> f(1,1) = 0
    n = 3  -> f(1,1,1) = 0
    n > 3  -> f(1,...,1) = 0
    """
    x = np.asarray_chkfinite(x)
    n = len(x)
    
    terms = np.zeros(n-1)
    for i in range(n-1):
        terms[i] = 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        
    return np.sum(terms)

def F81(x):  # Lévi Function N.13
    """
    Lévi Function N.13
    f(x,y) = sin^2(3πx) + (x-1)^2(1 + sin^2(3πy)) + (y-1)^2(1 + sin^2(2πy))
    Domain: Usually [-10, 10]
    Global minimum at (1,1) with f(1,1) = 0
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]  # x and y in the formula
    
    term1 = np.sin(3 * np.pi * x1)**2
    term2 = (x1 - 1)**2 * (1 + np.sin(3 * np.pi * x2)**2)
    term3 = (x2 - 1)**2 * (1 + np.sin(2 * np.pi * x2)**2)
    
    return term1 + term2 + term3

def F82(x):  # Simionescu Function
    """
    Simionescu Function (Constrained)
    f(x,y) = 0.1xy
    Subject to: x^2 + y^2 <= [r_T + r_S cos(n arctan(x/y))]^2
    Domain: -1.25 <= x,y <= 1.25
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    
    # Parameters
    r_T = 1
    r_S = 0.2
    n = 8
    
    # Constraint check
    angle = np.arctan2(x1, x2)
    r = r_T + r_S * np.cos(n * angle)
    if x1**2 + x2**2 > r**2:
        return 1e10  # Large penalty value instead of inf
    
    return 0.1 * x1 * x2

def F83(x):  # Gomez and Levy Function
    """
    Gomez and Levy Function (Constrained)
    Domain: -1 <= x <= 0.75, -1 <= y <= 1
    """
    x = np.asarray_chkfinite(x)
    x1, x2 = x[0], x[1]
    
    # Constraint check
    constraint = -np.sin(4*np.pi*x1) + 2*(np.sin(2*np.pi*x2))**2
    if constraint > 1.5:
        return 1e10  # Large penalty value instead of inf
    
    # Objective function
    term1 = 4 * x1**2 - 2.1 * x1**4 + (1/3) * x1**6
    term2 = x1 * x2
    term3 = -4 * x2**2 + 4 * x2**4
    return term1 + term2 + term3

def F84(x):  # Keane's Bump Function
    """
    Keane's Bump Function (Constrained)
    Domain: 0 < x_i < 10
    """
    x = np.asarray_chkfinite(x)
    m = len(x)
    
    # Constraint checks
    if 0.75 - np.prod(x) >= 0 or np.sum(x) - 7.5*m >= 0:
        return 1e10  # Large penalty value instead of inf
    
    cos4_sum = np.sum(np.cos(x)**4)
    cos2_prod = np.prod(np.cos(x)**2)
    ix2_sum = np.sum(np.arange(1, m+1) * x**2)
    
    numerator = np.abs(cos4_sum - 2 * cos2_prod)
    denominator = np.sqrt(ix2_sum)
    
    return -numerator/denominator

def ProblemsDetails(a):
    param = {
        "F1": ["F1", "Beale", -4.5, 4.5, 2],
        "F2": ["F2", "Easom", -10, 10, 2],
        "F3": ["F3", "Matyas", -10, 10, 2],
        "F4": ["F4", "Powell", -4, 5, 24],
        "F5": ["F5", "Schaffer No.1", -100, 100, 2],
        "F6": ["F6", "Schaffer No.3", -100, 100, 2],
        "F7": ["F7", "Schaffer No.4", -100, 100, 2],
        "F8": ["F8", "Zakharov", -5, 10, 30],
        "F9": ["F9", "Quartic", -1.28, 1.28, 30],
        "F10": ["F10", "Schwefel 2.21", -100, 100, 30],
        "F11": ["F11", "Schwefel 2.22", -10, 10, 30],
        "F12": ["F12", "Sphere", -100, 100, 30],
        "F13": ["F13", "Step2", -100, 100, 30],
        "F14": ["F14", "Step Integer", -5.12, 5.12, 5],
        "F15": ["F15", "Sum Squares", -10, 10, 30],
        "F16": ["F16", "Ackley", -32, 32, 30],
        "F17": ["F17", "Bohachevsky No.2", -10, 10, 2],
        "F18": ["F18", "Bohachevsky No.3", -100, 100, 2],
        "F19": ["F19", "Cross-in-Tray", -10, 10, 2],
        "F20": ["F20", "Griewank", -600, 600, 30],
        "F21": ["F21", "Goldstein-Price", -2, 2, 2],
        "F22": ["F22", "Hartman 3", 0, 1, 3],
        "F23": ["F23", "Hartman 6", 0, 1, 6],
        "F24": ["F24", "Penalized No.1", -50, 50, 30],
        "F25": ["F25", "Penalized No.2", -50, 50, 30],
        "F26": ["F26", "Perm", -4, 4, 4],
        "F27": ["F27", "Power Sum", 0, 4, 4],
        "F28": ["F28", "Shubert", -10, 10, 2],
        "F29": ["F29", "Alpine No.1", -10, 10, 30],
        "F30": ["F30", "Bohachevsky No.1", -100, 100, 2],
        "F31": ["F31", "Booth", -10, 10, 2],
        "F32": ["F32", "Branin", -5, 5, 2],
        "F33": ["F33", "Michalewicz 2", 0, np.pi, 2],
        "F34": ["F34", "Michalewicz 5", 0, np.pi, 5],
        "F35": ["F35", "Michalewicz 10", 0, np.pi, 10],
        "F36": ["F36", "Rastrigin", -5.12, 5.12, 30],
        "F37": ["F37", "Weierstrass", -0.5, 0.5, 30],
        "F38": ["F38", "Rotated Ackley", -32, 32, 30],
        "F39": ["F39", "Rotated Rastrigin", -5.12, 5.12, 30],
        "F40": ["F40", "Katsuura", -5, 5, 30],
        "F41": ["F41", "Whitley", -10, 10, 30],
        "F42": ["F42", "Salomon", -100, 100, 30],
        "F43": ["F43", "Langermann", 0, 10, 2],
        "F44": ["F44", "Kowalik", -5, 5, 4],
        "F45": ["F45", "BartelsConn", -500, 500, 2],
        "F46": ["F46", "Bird", -2*np.pi, 2*np.pi, 2],
        "F47": ["F47", "Box-Betts", 0.9, 11.2, 3],
        "F48": ["F48", "Bukin", [-15, -3], [-5, 3], 2],
        "F49": ["F49", "ThreeHumpCamel", -5, 5, 2],
        "F50": ["F50", "SixHumpCamel", -5, 5, 2],
        "F51": ["F51", "Chichinadze", -30, 30, 2],
        "F52": ["F52", "Cosine Mixture", -1, 1, 2],
        "F53": ["F53", "Csendes", -1, 1, 30],
        "F54": ["F54", "Cube", -10, 10, 2],
        "F55": ["F55", "Watson", -10, 10, 6],
        "F56": ["F56", "Xin-She Yang First", -5, 5, 30],
        "F57": ["F57", "Xin-She Yang Second", -2*np.pi, 2*np.pi, 30],
        "F58": ["F58", "Xin-She Yang Third", -20, 20, 30],
        "F59": ["F59", "Xin-She Yang Fourth", -10, 10, 30],
        "F60": ["F60", "Xin-She Yang Fifth", -10*np.pi, 10*np.pi, 30],
        "F61": ["F61", "Wood", -4, 4, 4],
        "F62": ["F62", "Vincent", 0.25, 10, 30],
        "F63": ["F63", "Egg Holder", -512, 512, 30],
        "F64": ["F64", "Noisy Quartic", -1.28, 1.28, 30],
        "F65": ["F65", "Modified Schwefel", -100, 100, 30],
        "F66": ["F66", "Alpine Modified", -10, 10, 30],
        "F67": ["F67", "Tripod", -100, 100, 2],
        "F68": ["F68", "Ursem", [-2.5, -2], [3, 2], 2],
        "F69": ["F69", "Ursem Waves", [-0.9, -1.2], [1.2, 1.2], 2],
        "F70": ["F70", "TestTube Holder", -10, 10, 2],
        "F71": ["F71", "Trecanni", -5, 5, 2],
        "F72": ["F72", "Trid", -6**2, 6**2, 2],
        "F73": ["F73", "Trefethen", -10, 10, 2],
        "F74": ["F74", "Styblinski-Tang", -5, 5, 30],
        "F75": ["F75", "First Holder Table", -10, 10, 2],
        "F76": ["F76", "Second Holder Table", -10, 10, 2],
        "F77": ["F77", "Carrom Table", -10, 10, 2],
        "F78": ["F78", "Shekel", 0, 10, 2],
        "F79": ["F79", "Trigonometric", 0, np.pi, 30],
        "F80": ["F80", "Rosenbrock", -2.048, 2.048, 30],
        "F81": ["F81", "Lévi N.13", -10, 10, 2],
        "F82": ["F82", "Simionescu", [-1.25, -1.25], [1.25, 1.25], 2],
        "F83": ["F83", "Gomez and Levy", [-1, -1], [0.75, 1], 2],
        "F84": ["F84", "Keane's Bump", [0, 0], [10, 10], 2]
    }
    return param.get(a, "nothing")

def get_single_objective_functions(dim):
    functions = []
    
    for i in range(1, 85):
        func_name = f"F{i}"
        details = ProblemsDetails(func_name)
        
        if details != "nothing":
            _, func_full_name, lb, ub, func_dim = details
            actual_dim = func_dim if func_dim != 30 else dim
            
            func = globals()[func_name]
            
            # Special handling for functions with different bounds per dimension
            if func_name in ["F48", "F68", "F69", "F83"]:
                bounds = np.array([lb, ub])  # Use the provided bounds directly
            else:
                # For functions with same bounds for all dimensions
                if isinstance(lb, (list, np.ndarray)):
                    bounds = np.array([lb, ub])  # Use provided bounds array
                else:
                    bounds = np.array([[lb, ub]] * actual_dim)
                
            functions.append((func_name, func_full_name, func, bounds, actual_dim))
            
    return functions

""" def run_single_objective_optimization(func_name, func_full_name, func, bounds, dim, num_particles, max_iter, plot_dir):
    print(f"\n{'='*50}")
    print(f"Optimizing {func_name}: {func_full_name}")
    print(f"{'='*50}")

    progress_direction = np.ones(dim)
    optimizer = DALO(func, dim, bounds, progress_direction, num_particles, max_iter)

    best_value, best_position = optimizer.optimize()

    print(f"Best Value: {best_value:.6e}")
    print(f"Best Position: {best_position}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iter + 1), optimizer.convergence_curve)
    plt.title(f"{func_name}: {func_full_name} - Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"{func_name}_{func_full_name.replace(' ', '_')}_convergence.png"))
    plt.close()

    return best_value, best_position """

def plot_single_objective_results(results, plot_dir):
    function_names = list(results.keys())
    best_values = [results[name][0] for name in function_names]

    plt.figure(figsize=(16, 8))
    bars = plt.bar(range(len(function_names)), best_values)
    plt.title("Best Values for Single-Objective Functions")
    plt.xlabel("Function Name")
    plt.ylabel("Best Value (log scale)")
    plt.yscale('log')
    plt.xticks(range(len(function_names)), [f"{name.split(':')[0]}\n{name.split(':')[1]}" for name in function_names], rotation=90)
    
    # Add value labels on top of each bar
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                 f'{height:.2e}',
                 ha='center', va='bottom', rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "single_objective_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()