#ifndef EASY_LIALG_MATRIX_INV
#define EASY_LIALG_MATRIX_INV

#include "./basic.h"
#include "./blas.h"
#include "./equation.h"

namespace EASYLINALG
{
    // https://www.youtube.com/watch?v=J8dSwvPfEc4
    // A_inv = (1/det(A)) adj (A)
    template <typename T>
    LIAG_FUNC_MACRO bool SymmInvertMatrixInner(const Matrix<T, 3, 3> &m, Matrix<T, 3, 3> &inv_m)
    {
        inv_m[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1];

        inv_m[1][0] = m[1][2] * m[2][0] - m[1][0] * m[2][2];

        inv_m[2][0] = m[1][0] * m[2][1] - m[1][1] * m[2][0];

        inv_m[0][1] = m[0][2] * m[2][1] - m[0][1] * m[2][2];

        inv_m[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0];

        inv_m[2][1] = m[0][1] * m[2][0] - m[0][0] * m[2][1];

        inv_m[0][2] = m[0][1] * m[1][2] - m[0][2] * m[1][1];

        inv_m[1][2] = m[0][2] * m[1][0] - m[0][0] * m[1][2];

        inv_m[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0];

        double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        if (det == 0)
        {
            printf("singular matrix\n");
            // matrix_show(m);
            return false;
        }

        det = 1.0 / det;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                inv_m[i][j] = inv_m[i][j] * det;
            }
        }

        return true;
    }

    template <typename T>
    LIAG_FUNC_MACRO bool SymmInvertMatrixInner(const Matrix<T, 4, 4> &m, Matrix<T, 4, 4> &inv_m)
    {
        inv_m[0][0] = m[1][1] * m[2][2] * m[3][3] -
                      m[1][1] * m[2][3] * m[3][2] -
                      m[2][1] * m[1][2] * m[3][3] +
                      m[2][1] * m[1][3] * m[3][2] +
                      m[3][1] * m[1][2] * m[2][3] -
                      m[3][1] * m[1][3] * m[2][2];

        inv_m[1][0] = -m[1][0] * m[2][2] * m[3][3] +
                      m[1][0] * m[2][3] * m[3][2] +
                      m[2][0] * m[1][2] * m[3][3] -
                      m[2][0] * m[1][3] * m[3][2] -
                      m[3][0] * m[1][2] * m[2][3] +
                      m[3][0] * m[1][3] * m[2][2];

        inv_m[2][0] = m[1][0] * m[2][1] * m[3][3] -
                      m[1][0] * m[2][3] * m[3][1] -
                      m[2][0] * m[1][1] * m[3][3] +
                      m[2][0] * m[1][3] * m[3][1] +
                      m[3][0] * m[1][1] * m[2][3] -
                      m[3][0] * m[1][3] * m[2][1];

        inv_m[3][0] = -m[1][0] * m[2][1] * m[3][2] +
                      m[1][0] * m[2][2] * m[3][1] +
                      m[2][0] * m[1][1] * m[3][2] -
                      m[2][0] * m[1][2] * m[3][1] -
                      m[3][0] * m[1][1] * m[2][2] +
                      m[3][0] * m[1][2] * m[2][1];

        inv_m[0][1] = -m[0][1] * m[2][2] * m[3][3] +
                      m[0][1] * m[2][3] * m[3][2] +
                      m[2][1] * m[0][2] * m[3][3] -
                      m[2][1] * m[0][3] * m[3][2] -
                      m[3][1] * m[0][2] * m[2][3] +
                      m[3][1] * m[0][3] * m[2][2];

        inv_m[1][1] = m[0][0] * m[2][2] * m[3][3] -
                      m[0][0] * m[2][3] * m[3][2] -
                      m[2][0] * m[0][2] * m[3][3] +
                      m[2][0] * m[0][3] * m[3][2] +
                      m[3][0] * m[0][2] * m[2][3] -
                      m[3][0] * m[0][3] * m[2][2];

        inv_m[2][1] = -m[0][0] * m[2][1] * m[3][3] +
                      m[0][0] * m[2][3] * m[3][1] +
                      m[2][0] * m[0][1] * m[3][3] -
                      m[2][0] * m[0][3] * m[3][1] -
                      m[3][0] * m[0][1] * m[2][3] +
                      m[3][0] * m[0][3] * m[2][1];

        inv_m[3][1] = m[0][0] * m[2][1] * m[3][2] -
                      m[0][0] * m[2][2] * m[3][1] -
                      m[2][0] * m[0][1] * m[3][2] +
                      m[2][0] * m[0][2] * m[3][1] +
                      m[3][0] * m[0][1] * m[2][2] -
                      m[3][0] * m[0][2] * m[2][1];

        inv_m[0][2] = m[0][1] * m[1][2] * m[3][3] -
                      m[0][1] * m[1][3] * m[3][2] -
                      m[1][1] * m[0][2] * m[3][3] +
                      m[1][1] * m[0][3] * m[3][2] +
                      m[3][1] * m[0][2] * m[1][3] -
                      m[3][1] * m[0][3] * m[1][2];

        inv_m[1][2] = -m[0][0] * m[1][2] * m[3][3] +
                      m[0][0] * m[1][3] * m[3][2] +
                      m[1][0] * m[0][2] * m[3][3] -
                      m[1][0] * m[0][3] * m[3][2] -
                      m[3][0] * m[0][2] * m[1][3] +
                      m[3][0] * m[0][3] * m[1][2];

        inv_m[2][2] = m[0][0] * m[1][1] * m[3][3] -
                      m[0][0] * m[1][3] * m[3][1] -
                      m[1][0] * m[0][1] * m[3][3] +
                      m[1][0] * m[0][3] * m[3][1] +
                      m[3][0] * m[0][1] * m[1][3] -
                      m[3][0] * m[0][3] * m[1][1];

        inv_m[3][2] = -m[0][0] * m[1][1] * m[3][2] +
                      m[0][0] * m[1][2] * m[3][1] +
                      m[1][0] * m[0][1] * m[3][2] -
                      m[1][0] * m[0][2] * m[3][1] -
                      m[3][0] * m[0][1] * m[1][2] +
                      m[3][0] * m[0][2] * m[1][1];

        inv_m[0][3] = -m[0][1] * m[1][2] * m[2][3] +
                      m[0][1] * m[1][3] * m[2][2] +
                      m[1][1] * m[0][2] * m[2][3] -
                      m[1][1] * m[0][3] * m[2][2] -
                      m[2][1] * m[0][2] * m[1][3] +
                      m[2][1] * m[0][3] * m[1][2];

        inv_m[1][3] = m[0][0] * m[1][2] * m[2][3] -
                      m[0][0] * m[1][3] * m[2][2] -
                      m[1][0] * m[0][2] * m[2][3] +
                      m[1][0] * m[0][3] * m[2][2] +
                      m[2][0] * m[0][2] * m[1][3] -
                      m[2][0] * m[0][3] * m[1][2];

        inv_m[2][3] = -m[0][0] * m[1][1] * m[2][3] +
                      m[0][0] * m[1][3] * m[2][1] +
                      m[1][0] * m[0][1] * m[2][3] -
                      m[1][0] * m[0][3] * m[2][1] -
                      m[2][0] * m[0][1] * m[1][3] +
                      m[2][0] * m[0][3] * m[1][1];

        inv_m[3][3] = m[0][0] * m[1][1] * m[2][2] -
                      m[0][0] * m[1][2] * m[2][1] -
                      m[1][0] * m[0][1] * m[2][2] +
                      m[1][0] * m[0][2] * m[2][1] +
                      m[2][0] * m[0][1] * m[1][2] -
                      m[2][0] * m[0][2] * m[1][1];

        double det = m[0][0] * inv_m[0][0] +
                     m[0][1] * inv_m[1][0] +
                     m[0][2] * inv_m[2][0] +
                     m[0][3] * inv_m[3][0];

        if (det == 0)
        {
            printf("singular matrix for 4by4\n");
            // matrix_show(m);
            return false;
        }

        det = 1.0 / det;

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                inv_m[i][j] = inv_m[i][j] * det;
            }
        }

        return true;
    }

    // for small matrix, we use the direact inverse to solve the linear equation
    // for large matrix, we need to use other method such as qr things to solve
    // TODO, checking singularity for inverting the matrix
    // refer to https://inst.eecs.berkeley.edu/~ee127/sp21/livebook/l_lineqs_solving.html
    // computing the A_inv by solving A*A_inv=I computing each colum of A_inv each time
    template <typename T, uint Size>
    LIAG_FUNC_MACRO bool SymmInvertMatrixInner(const Matrix<T, Size, Size> &m, Matrix<T, Size, Size> &inv_m)
    {
        // TODO return false for singular matrix

        // QR decomposition of matrix
        Matrix<T, Size, Size> Q;
        Matrix<T, Size, Size> R;
        Householder(m, Q, R);

        // puts("inv q r");
        // Q.Show();
        // R.Show();

        // Q*R*x=b
        // R*x = Q_t*b
        // b is each colum of the I matrix
        for (uint j = 0; j < Size; j++)
        {
            Vec<double, Size> b(0.0);
            b[j] = 1.0;

            // compute the transpose of Q
            Matrix<T, Size, Size> Q_t = Q;
            Q_t.Transpose();

            // puts("Q_t");
            // Q_t.Show();

            // compute Qt*b
            Vec<T, Size> Q_tb = MVMultiply(Q_t, b);
            // puts("Q_tb");
            // Q_t.Show();

            // using back substition to solve R*x=Q_t*b
            Vec<T, Size> x = SymmBackSubstitution(R, Q_tb);
            // printf("colm %d \n for x\n", j);
            // x.Show();

            // putting associating colum into the jth colum of inv_m matrix
            for (uint i = 0; i < Size; i++)
            {
                inv_m[i][j] = x[i];
            }
        }
        // inv_m.Show();
        return true;
    }

    template <typename T,
              uint Size>
    LIAG_FUNC_MACRO bool SymmInvertMatrix(const Matrix<T, Size, Size> &A, Matrix<T, Size, Size> &AInv)
    {
        // it seems there is issue
        // if we implement different function according to the value of Size
        SymmInvertMatrixInner(A, AInv);
        return true;
    }

}

#endif