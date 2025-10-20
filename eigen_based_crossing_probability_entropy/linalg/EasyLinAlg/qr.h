#ifndef EASY_LIALG_QR
#define EASY_LIALG_QR

#include "./basic.h"
#include "./blas.h"

namespace EASYLINALG
{
    template <typename T,
              uint NumRow,
              uint NumCol>
    LIAG_FUNC_MACRO void Householder(
        const Matrix<T, NumRow, NumCol> &A,
        Matrix<T, NumRow, NumCol> &Q,
        Matrix<T, NumRow, NumCol> &R)
    {
        // current code only works for the symetry matrix
        assert(NumRow == NumCol);
        const int msize = NumRow;
        // Matrix<T, NumRow, NumCol> MList[msize];
        Matrix<T, NumRow, NumCol> MTemp;

        // operator both for const and non-const
        // MList[0] = A;
        MTemp = A;

        R = A;
        double ANorm;

        Vec<T, msize> Ve;
        Ve.InitZero();
        Vec<T, msize> Va;
        Va.InitZero();

        for (uint k = 1; k < msize; k++)
        {
            auto HInit = R.GetMinor(k - 1);
            // printf("hhd iter%d\n", k);
            // HInit.Show();

            Va = HInit.GetColum(k - 1);

            // puts("a colum");
            // Va.Show();

            ANorm = Va.GetNorm();

            // printf("a norm %f\n", ANorm);

            if (A[k - 1][k - 1] > 0)
                ANorm = -ANorm;

            for (uint i = 0; i < msize; i++)
                Ve[i] = (i == (k - 1)) ? 1 : 0;

            // puts("ve is");
            // Ve.Show();
            // puts("va is");
            // Va.Show();

            // Ve = ANorm*Ve+Va
            Ve = AXPY(ANorm, Ve, Va);
            // puts("after vmadd v vec");
            // Ve.Show();

            // TODO, only do this when norm is not zero
            if (Ve.GetNorm() > 0.00001)
            {
                Ve.Div(Ve.GetNorm());
            }

            // puts("after vdiv v vec");
            // Ve.Show();

            // I-v*v
            // MList[k] = IMINUSVVT(Ve);
            MTemp = IMINUSVVT(Ve);
            // puts("mat_list[k] H is");
            // MTemp.Show();

            // update Q
            if (k == 1)
            {
                // Q = MList[k];
                Q = MTemp;
            }
            else
            {
                // Q = MatrixMultiply(Q, MList[k]);
                Q = MMMultiply(Q, MTemp);
            }

            // update R
            // puts("before updating R");
            // puts("MList[k]");
            // MTemp.Show();

            // puts("R");
            // R.Show();

            // R = MatrixMultiply(MList[k], R);
            R = MMMultiply(MTemp, R);

            // puts("update R is");
            // R.Show();
        }
    }
}

#endif