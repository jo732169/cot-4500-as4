import numpy as np

#Q1 Jacobi Iterative Method
def JacobiIterativeMethod(MaxIterations=20, tolerance=.01, k=0):

    #populate Augmented Matrix using p. 454
    A = [[10, -1, 2, 0, 6], 
         [-1, 11, -1, 3, 25], 
         [2, -1, 10, -1, -11], 
         [0, 3, -1, 8, 15]]
    
    num = len(A)

    XOi = np.array([0]*num, dtype=float)
    xi = np.array([0]*num, dtype=float)
 
    #while loop until procedure succesful (iteration condition met)
    while k <= MaxIterations:
        i=1
        for i in range(num):
            sum = 0
            for j in range(num):
                if j != i:
                    sum += A[i][j] * XOi[j]

            xi[i] = (1/A[i][i])*(A[i][len(A)]-sum)

        #linear algebra solver to determine x^k and x^k-1 tolerance
        if np.linalg.norm(xi-XOi) < tolerance:
            print(xi)
            return
        k +=1
        for i in range(num):
            #Set Xoi=xi
            XOi[i] = xi[i]
    
    print("Max number of iterations exceeded. Procedure succesful.")
    return

#Q2 : Gauss-Seidel IterativeMethod
def GaussSeidelIterativeMethod(MaxIterations=20, tolerance=.01, k=0):

    #populate Augmented Matrix
    A = [[10, -1, 2, 0, 6], 
         [-1, 11, -1, 3, 25], 
         [2, -1, 10, -1, -11], 
         [0, 3, -1, 8, 15]]
   
    num= len(A)

    XOi = np.array([0] * num, dtype=float)
    xi= np.array([0] * num, dtype=float)

    #while loop until procedure succesful (iteration condition met)
    while k <= MaxIterations:
        i=1
        for i in range(num):
            sum = 0
            for j in range(i):
                sum += A[i][j] * xi[j]
                for j in range(i+1, num):
                 sum += A[i][j] * XOi[j]
            
            xi[i] = (1/A[i][i])*(A[i][len(A)]-sum)

        #linear algebra solver to determine x^k and x^k-1 tolerance
        if np.linalg.norm(xi-XOi) < tolerance:
            print(xi)
            return
        k += 1
        for i in range(num):
            #Set Xoi=xi
            XOi[i] = xi[i]
    
    print("Max number of iterations exceeded. Procedure succesful.")
    return

# Q3: SOR Method
def SORMethod(MaxIterations=20, tolerance=.01, k=0, w=1.25):

    A = [[10, -1, 2, 0, 6], 
         [-1, 11, -1, 3, 25], 
         [2, -1, 10, -1, -11], 
         [0, 3, -1, 8, 15]]
   
    n = len(A)


    XOi = np.array([0] * n, dtype=float)
    xi = np.array([1] * n, dtype=float)

    #while loop until procedure succesful (iteration condition met)
    while k <= MaxIterations:
        for i in range(n):
            sum = 0
            for j in range(i):
                sum += A[i][j] * xi[j]
            for j in range(i+1, n):
                sum += A[i][j] * XOi[j]

            xi[i] = (1 / A[i][i]) * w * (A[i][len(A)] - sum) + ((1-w) * XOi[i])

        #linear algebra solver to determine x^k and x^k-1 tolerance
        if np.linalg.norm(xi-XOi) < tolerance:
            print(xi)
            return
        k += 1
        for i in range(n):
            #Set Xoi=xi
            XOi[i] = xi[i]
    
    print("Maximum number of iterations exceeded. Procedure succesful.")
    return

# Q4: Iterative Refinement 
def Gaussian(MatA):
    n = len(MatA)
    for i in range(1,n):
        p=i

        if (p != i):
            switch = MatA[i-1]
            MatA[i-1] = MatA[p-1]
            MatA[p-1] = switch
        
        for j in range(i+1, n+1):
            m = MatA[j-1][i-1] / MatA[i-1][i-1]
            MatA[j-1] = MatA[j-1] - m * MatA[i-1]
            x = np.zeros(n)
        
    if MatA[n-1][n-1] == 0:
      return 0
    

    x[n-1] = MatA[n-1][n] / MatA[n-1][n-1]

    for i in range(n-1, 0, -1):
        sum = 0
        for j in range(i+1, n+1):
            sum += MatA[i-1][j-1]*x[j-1]
        x[i-1] = (MatA[i-1][n] - sum)/MatA[i-1][i-1]
    
    return x

def IterativeRefinement(MaxIterations=100, tolerance=.00005, k=0, t=5):
    
    #Matrix A
    MatA = [[3.3330, 15920, -10.333], 
         [2.2220, 16.710, 9.6120], 
         [1.5611, 5.1791, 1.6852]]
    
    #Matrix B
    B = np.array([15913, 
                  28.544, 
                  8.4254])
    num=len(MatA)
    COND=None

    AugmentedA = np.array([[3.3330, 15920, -10.333, 15913], 
         [2.2220, 16.710, 9.6120, 28.544], 
        [1.5611, 5.1791, 1.6852, 8.4254]])


    Xo = np.zeros(num, dtype=float)
    XOi = np.zeros(num, dtype=float)


    while k <= MaxIterations:
        
        xi = Gaussian(AugmentedA)
      
        for i in range(num):
            sum = 0
            for j in range(num):
                sum += MatA[i][j] * xi[j]
            
            Xo[i] = B[i] - sum

        
        Append = np.zeros((num, num+1))
        for i in range(len(MatA)):
            for j in range(len(MatA[i])):
                Append[i][j] = MatA[i][j]
            Append[i][num] = Xo[i]

      
        solve = Gaussian(Append)

       
        for i in range(num):
            
            XOi[i] = xi[i] + solve[i]

        if k == 1:
            COND = (np.linalg.norm(y) / np.linalg.norm(XOi)) * (5*t)

        #Determine if x and X0i are within tolereance
        if np.linalg.norm(xi-XOi) < tolerance:
            print(XOi)
            print(COND)
            return
      
        for i in range(num):
            xi[i] = XOi[i]

    return




#Print output 
if __name__ == '__main__':
    print("Q1: Jacobi Iterative")
    JacobiIterativeMethod()
    print("Q2: Guass-Seidel Iterative Method")
    GaussSeidelIterativeMethod()
    print("Q3: SOR Method")
    SORMethod()
    print("Q4: Iterative Refinement")
    IterativeRefinement()