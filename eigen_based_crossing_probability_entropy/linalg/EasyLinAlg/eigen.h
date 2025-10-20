// eigen value and eigen vectors

#ifndef EASY_LIALG_EIGEN
#define EASY_LIALG_EIGEN

#include "./basic.h"
#include "./blas.h"
#include "./qr.h"
#include "./matrixinv.h"

namespace EASYLINALG
{
    template <typename T, uint Size>
    LIAG_FUNC_MACRO void SymmEigenValuesShift(const Matrix<T, Size, Size> &A, double tol, int maxIter, Vec<T, Size> &eigenValues)
    {
        // refer to https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition
        Matrix<T, Size, Size> Ak = A;

        // mat_t qq = matrix_new_eye(x->m, x->m);
        Matrix<T, Size, Size> QQ;
        QQ.InitEye();

        Matrix<T, Size, Size> Eye;
        Eye.InitEye();

        Matrix<T, Size, Size> R, Q;
        //int actualIter = 0;
        for (int i = 0; i < maxIter; ++i)
        {
            //actualIter = i;
            // s_k is the last item of the first diagonal
            T s = Ak[Size - 1][Size - 1];
            Matrix<T, Size, Size> smulI = MSCALE(s, Eye);
            Householder(Ak - smulI, Q, R);
            // printf("iter i %d \nR\n", i);
            // R.Show();
            // puts("Q");
            // Q.Show();

            // addinng smulI back
            Ak = MMMultiply(R, Q) + smulI;
            QQ = MMMultiply(QQ, Q);

            // Ak.Show();
            if (Ak.IsUpperTriangular(tol))
            {
                break;
            }
        }

        //std::cout << "actual iter is " << actualIter << std::endl;

        for (uint i = 0; i < Size; i++)
        {
            if (fabs(Ak[i][i] - 0) < tol)
            {
                eigenValues[i] = 0.0;
            }
            else
            {
                eigenValues[i] = Ak[i][i];
            }
        }
    }

    template <typename T, uint Size>
    LIAG_FUNC_MACRO void SymmEigenValues(const Matrix<T, Size, Size> &A, double tol, int maxIter, Vec<T, Size> &eigenValues)
    {
        // refer to https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition
        // mat_t ak = *x;
        Matrix<T, Size, Size> Ak = A;

        // mat_t qq = matrix_new_eye(x->m, x->m);
        Matrix<T, Size, Size> QQ;
        QQ.InitEye();

        Matrix<T, Size, Size> R, Q;
        for (int i = 0; i < maxIter; ++i)
        {

            Householder(Ak, Q, R);
            // printf("iter i %d \nR\n", i);
            // R.Show();
            // puts("Q");
            // Q.Show();

            Ak = MMMultiply(R, Q);
            QQ = MMMultiply(QQ, Q);

            if (Ak.IsUpperTriangular(tol))
            {
                break;
            }
        }

        for (uint i = 0; i < Size; i++)
        {
            if (fabs(Ak[i][i] - 0) < tol)
            {
                eigenValues[i] = 0.0;
            }
            else
            {
                eigenValues[i] = Ak[i][i];
            }
        }
    }

    // given an matrix and associated eigen value and then compute associated eigen vectors
    // the size represent the length of associated eigen vector
    template <typename T, uint Size>
    LIAG_FUNC_MACRO Vec<T, Size> ComputeEigenVectors(const Matrix<T, Size, Size> &A, const T &eigenValue, uint maxIter)
    {
        Vec<T, Size> eigenVectors(0);
        Matrix<T, Size, Size> A_minus_lambda_i;
        Matrix<T, Size, Size> A_minus_lambda_inv;
        if (fabs(eigenValue - 0.0) < 0.00001)
        {
            // the associated eigen vector is all zero
            return eigenVectors;
        }
        // do the inverse iteration
        double lambda = eigenValue + 0.000001;
        for (uint ii = 0; ii < Size; ii++)
        {
            for (uint jj = 0; jj < Size; jj++)
            {
                if (ii == jj)
                {
                    A_minus_lambda_i[ii][jj] = A[ii][jj] - lambda;
                }
                else
                {
                    A_minus_lambda_i[ii][jj] = A[ii][jj];
                }
            }
        }

        bool invertok = SymmInvertMatrix(A_minus_lambda_i, A_minus_lambda_inv);
        assert(invertok == true);

        Vec<T, Size> bPrev;
        Vec<T, Size> bCurr;
        // init as 1
        for (uint i = 0; i < Size; i++)
        {
            bPrev[i] = 1.0;
        }
        uint iterNum = 0;
        while (iterNum < maxIter)
        {
            bCurr = MVMultiply(A_minus_lambda_inv, bPrev);
            // puts("check mmv");
            // bCurr.Show();
            bCurr.Normalize();
            // puts("check norm");
            // bCurr.Show();
            if (bCurr.Equal(bPrev))
            {
                // results convert
                break;
            }
            bPrev = bCurr;
            iterNum++;
        }
        // assign results to eigen vectors
        for (uint i = 0; i < Size; i++)
        {
            eigenVectors[i] = bCurr[i];
        }
        return eigenVectors;
    }

    // using inverse itertaion to compute the eigen vectors
    // https://en.wikipedia.org/wiki/Inverse_iteration
    template <typename T, uint Size>
    LIAG_FUNC_MACRO Matrix<T, Size, Size> SymmEigenVectors(const Matrix<T, Size, Size> &A, const Vec<T, Size> &eigenValues, uint maxIter)
    {

        Matrix<T, Size, Size> eigenVectors;
        Matrix<T, Size, Size> A_minus_lambda_i;
        Matrix<T, Size, Size> A_minus_lambda_inv;
        // go through each eigen values
        for (uint j = 0; j < Size; j++)
        {
            if (fabs(eigenValues[j] - 0.0) < 0.00001)
            {
                // the associated eigen vector is all zero
                continue;
            }

            double lambda = eigenValues[j] + 0.000001;
            for (uint ii = 0; ii < Size; ii++)
            {
                for (uint jj = 0; jj < Size; jj++)
                {
                    if (ii == jj)
                    {
                        A_minus_lambda_i[ii][jj] = A[ii][jj] - lambda;
                    }
                    else
                    {
                        A_minus_lambda_i[ii][jj] = A[ii][jj];
                    }
                }
            }

            // puts("a-lambda*I");:1
            // A_minus_lambda_i.Show();

            bool invertok = SymmInvertMatrix(A_minus_lambda_i, A_minus_lambda_inv);
            assert(invertok == true);

            // printf("eigen vec %f a-lambda*I inv\n", eigenValues[j]);
            // A_minus_lambda_inv.Show();

            Vec<T, Size> bPrev;
            Vec<T, Size> bCurr;
            // init as 1
            for (uint i = 0; i < Size; i++)
            {
                bPrev[i] = 1.0;
            }
            uint iterNum = 0;
            while (iterNum < maxIter)
            {
                bCurr = MVMultiply(A_minus_lambda_inv, bPrev);
                // puts("check mmv");
                // bCurr.Show();

                bCurr.Normalize();
                // puts("check norm");
                // bCurr.Show();

                if (bCurr.Equal(bPrev))
                {
                    // results convert
                    break;
                }
                bPrev = bCurr;
                iterNum++;
            }

            // assign to the matrix
            for (uint i = 0; i < Size; i++)
            {
                eigenVectors[i][j] = bCurr[i];
            }
        }
        return eigenVectors;
    }

    // The input matrix such as the covariance matrix
    // is supposed to be a symmetric positive definite matrix with all
    // eigen values larger than 0
    template <typename T, uint Size>
    LIAG_FUNC_MACRO Matrix<T, Size, Size> SymmEigenDecomposition(const Matrix<T, Size, Size> &A, double tol, int maxIter, bool useShift = false)
    {
        // solve eigen values
        Vec<T, Size> eigenValues;
        eigenValues.InitZero();
        if (useShift)
        {
            // the values shift converge faster than the naive eigen values case
            // but it is error prone for some test cases
            SymmEigenValuesShift(A, tol, maxIter, eigenValues);
        }
        else
        {
            SymmEigenValues(A, tol, maxIter, eigenValues);
        }

        // std::cout << "eigen values" << std::endl;
        // eigenValues.Show();

        // solve eigen vectors
        Matrix<T, Size, Size> eigenVactors;
        eigenVactors = SymmEigenVectors(A, eigenValues, maxIter);

        // create diagonal matrix
        Matrix<T, Size, Size> diag;
        for (uint i = 0; i < Size; i++)
        {

            if (eigenValues[i] < 0)
            {
                //TODO: handle case where there is 0 in the matrix, this might introduce negative eigen
                if (fabs(eigenValues[i]) < 0.001)
                {
                    // make sure all value is >0 and we can compute sqrt for it
                    // there are some numerical errors for computing the eigen values
                    eigenValues[i] = -eigenValues[i];
                }
                else
                {
                    // debug use
                    printf("eigen values are\n");
                    for (uint j = 0; j < Size; j++)
                    {
                        printf(" %f ", eigenValues[j]);
                    }
                    printf("the eigen value is supposed to be >=0\n");
                    assert(false);
                }
            }

            diag[i][i] = sqrt(eigenValues[i]);
        }

        Matrix<T, Size, Size> decompA;
        decompA = MMMultiply(eigenVactors, diag);
        return decompA;
    }
}

#endif
