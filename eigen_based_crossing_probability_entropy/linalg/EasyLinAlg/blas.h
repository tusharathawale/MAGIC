
// supporting https://www.gnu.org/software/gsl/doc/html/blas.html

#ifndef EASY_LIALG_BLAS
#define EASY_LIALG_BLAS

#include "./basic.h"

namespace EASYLINALG
{

// x=a*x
template <typename T, uint Size>
LIAG_FUNC_MACRO Matrix<T, Size, Size> MSCALE(const T v, const Matrix<T, Size, Size> &x)
{
    Matrix<T, Size, Size> Mat;
    for (uint i = 0; i < Size; i++)
    {
        for (uint j = 0; j < Size; j++)
        {
            Mat[i][j] = v * x[i][j];
        }
    }
    return Mat;
}

// m = I - 2* v v^T
template <typename T, uint Size>
LIAG_FUNC_MACRO Matrix<T, Size, Size> IMINUSVVT(Vec<T, Size> &v)
{
    // this member become the static const member?
    Matrix<T, Size, Size> Mat;
    for (uint i = 0; i < Size; i++)
        for (uint j = 0; j < Size; j++)
            Mat[i][j] = -2 * v[i] * v[j];
    // diagnal elements add 1
    for (uint i = 0; i < Size; i++)
        Mat[i][i] = Mat[i][i] + 1;

    return Mat;
}

template <typename T,
          uint NumRow,
          uint NumCol,
          uint NumInternal>
LIAG_FUNC_MACRO Matrix<T, NumRow, NumCol> MMMultiply(
    Matrix<T, NumRow, NumInternal> &leftFactor,
    Matrix<T, NumInternal, NumCol> &rightFactor)
{
    Matrix<T, NumRow, NumCol> result;

    for (uint i = 0; i < NumRow; i++)
    {
        for (uint j = 0; j < NumCol; j++)
        {
            // init specific position of z as zero
            result[i][j] = 0;
            for (uint k = 0; k < NumInternal; k++)
            {
                result[i][j] = result[i][j] + leftFactor[i][k] * rightFactor[k][j];
            }
        }
    }
    return result;
}

//assuming vector is 1*row, and the matrix is row*column
template <typename T,
          uint NumRow,
          uint NumCol>
LIAG_FUNC_MACRO Vec<T, NumCol> VMMultiply(const Vec<T, NumRow> inputV,const Matrix<T, NumRow, NumCol> &inputM)
{
    Vec<T, NumCol> result;
    result.InitZero();
    for (uint i = 0; i < NumCol; i++)
    {
        for (uint j = 0; j < NumRow; j++)
        {
            // for each row
            result[i] += inputV[j]*inputM[j][i];
        }
    }
    return result;
}

template <typename T,
          uint NumRow,
          uint NumCol>
LIAG_FUNC_MACRO Vec<T, NumRow> MVMultiply(const Matrix<T, NumRow, NumCol> &inputM, const Vec<T, NumCol> inputV)
{
    Vec<T, NumRow> result;
    result.InitZero();
    for (uint i = 0; i < NumRow; i++)
    {
        for (uint j = 0; j < NumCol; j++)
        {
            // for each row
            result[i] += inputM[i][j] * inputV[j];
        }
    }
    return result;
}

// matrix multiply vector plus vector
// A*U+M
template <typename T,
          uint NumRow,
          uint NumCol>
LIAG_FUNC_MACRO Vec<T, NumRow> MMVPV(
    Matrix<T, NumRow, NumCol> &A,
    Vec<T, NumCol> &U,
    Vec<T, NumRow> &M)
{
    Vec<T, NumRow> AUM;
    AUM.InitZero();
    // Be careful about it, init the vector to zero
    for (uint i = 0; i < NumRow; i++)
    {
        // adding vector into the matrix
        for (uint j = 0; j < NumCol; j++)
        {
            AUM[i] = AUM[i] + (A[i][j] * U[j]);
        }
        AUM[i] = AUM[i] + M[i];
    }

    return AUM;
}

// results = a*x + b
template <typename T, uint Size>
LIAG_FUNC_MACRO Vec<T, Size> AXPY(double a, Vec<T, Size> &x, Vec<T, Size> &b)
{
    Vec<T, Size> result;
    for (uint i = 0; i < Size; i++)
    {
        result[i] = a * x[i] + b[i];
    }
    return result;
}

// A is matrix, x and y are vectors
//  results =alpha*A*x+beta*y
template <typename T, uint Row, uint Col>
LIAG_FUNC_MACRO Vec<T, Row> DGEMV(const T alpha, const Matrix<T, Row, Col> &A, Vec<T, Col> &x, const T beta, const Vec<T, Row> &y)
{
    Vec<T, Row> result;
    result.InitZero();
    for (uint i = 0; i < Row; i++)
    {
        for (uint j = 0; j < Col; j++)
        {
            result[i] += A[i][j] * x[j];
        }
        result[i] = result[i] * alpha;
        result[i] = result[i] + beta * y[i];
    }
    return result;
}

}
#endif
