// Computing determinant of matrix
// for 2*2 3*3 and 4*4, we compute them direactly
// for large scale matrix, we use the QR decomposition

#ifndef EASY_LIALG_MATRIX_DET
#define EASY_LIALG_MATRIX_DET

#include "./basic.h"
#include "./blas.h"
#include "./eigen.h"

namespace EASYLINALG
{
    // 1> Assuming using QR approach, the det(A)=det(Q)*det(R), but it is hard to know the det(Q) ois +1 or -1

    // 2> determinant based on PLU decomposition

    // 3> determiant based on products of eigen values
    // input original mastrix, output the detminant value

    template <typename T, uint Size>
    LIAG_FUNC_MACRO double SymmDet3by3(const Matrix<T, Size, Size> &m)
    {
        double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        return det;
    }

    template <typename T, uint Size>
    LIAG_FUNC_MACRO double SymmDetByEigenValues(const Matrix<T, Size, Size> &A, double tol = 0.00001, int maxIter = 1000)
    {

        if (Size == 3)
        {
            return SymmDet3by3(A);
        }

        // LIAG_FUNC_MACRO void
        Vec<T, Size> eigenValues;
        SymmEigenValues(A, tol, maxIter, eigenValues);
        // return products of the eigenValues
        double det = 1.0;
        for (uint i = 0; i < Size; i++)
        {
            det *= eigenValues[i];
        }
        return det;
    }
}

#endif