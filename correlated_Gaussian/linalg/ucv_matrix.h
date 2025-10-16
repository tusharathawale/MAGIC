#ifndef UCV_MATRIX_H
#define UCV_MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <cassert>

#include <viskores/Types.h>

#ifdef VISKORES_CUDA
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#else
// using the std library
#include <random>
#endif // VISKORES_CUDA

namespace UCVMATH
{

    typedef struct
    {
        int m, n; // m is row, n is column
        double **v = NULL;
    } mat_t, *mat;

    typedef struct
    {
        int len;
        double *v = NULL;
    } vec_t, *vec;

    VISKORES_EXEC inline vec vec_new(int len)
    {
        vec v = (vec)malloc(sizeof(vec_t));
        assert(v != NULL);
        v->len = len;
        v->v = (double *)malloc(sizeof(double) * len);
        assert(v->v != NULL);
        for (int i = 0; i < len; i++)
        {
            v->v[i] = 0;
        }
        return v;
    }

    VISKORES_EXEC inline void vec_show(vec v)
    {

        for (int i = 0; i < v->len; i++)
        {
            printf("%f ", v->v[i]);
        }
        printf("\n");
    }

    VISKORES_EXEC inline void vec_delete(vec v)
    {
        if (v->v != NULL)
        {
            free(v->v);
        }
        if (v != NULL)
        {
            free(v);
        }
    }

    VISKORES_EXEC inline mat matrix_new(int m, int n)
    {
        mat x = (mat)malloc(sizeof(mat_t));
        assert(x != NULL);
        x->v = (double **)malloc(sizeof(double *) * m);
        assert(x->v != NULL);
        // the init memory are all zero
        // cuda does not like calloc, malloc then give zero to it manually
        // x->v[0] = (double *)calloc(sizeof(double), m * n);
        // continus m*n elements
        x->v[0] = (double *)malloc(sizeof(double) * m * n);
        assert(x->v[0] != NULL);
        for (int i = 0; i < m * n; i++)
        {
            x->v[0][i] = 0;
        }
        // set the position of the first pointer in each row
        for (int i = 0; i < m; i++)
            x->v[i] = x->v[0] + n * i;
        x->m = m;
        x->n = n;
        return x;
    }

    VISKORES_EXEC inline mat *matrix_new_array_entry(int m, int n, int array_num)
    {
        mat *m_array = (mat *)malloc(sizeof(mat) * array_num);
        assert(m_array != NULL);
        //for (int i = 0; i < array_num; i++)
        //{
        //    m_array[i] = matrix_new(m, n);
        //}
        return m_array;
    }

    VISKORES_EXEC inline mat matrix_new_eye(int m, int n)
    {
        mat x = (mat)malloc(sizeof(mat_t));
        x->v = (double **)malloc(sizeof(double *) * m);
        // cuda do not like calloc
        // x->v[0] = (double *)calloc(sizeof(double), m * n);
        x->v[0] = (double *)malloc(sizeof(double) * m * n);
        for (int i = 0; i < m * n; i++)
        {
            x->v[0][i] = 0;
        }
        for (int i = 0; i < m; i++)
        {
            x->v[i] = x->v[0] + n * i;
            x->v[i][i] = 1;
        }
        x->m = m;
        x->n = n;
        return x;
    }

    VISKORES_EXEC inline void matrix_delete(mat m)
    {
        free(m->v[0]);
        free(m->v);
        free(m);
    }

    VISKORES_EXEC inline void matrix_transpose(mat m)
    {
        for (int i = 0; i < m->m; i++)
        {
            for (int j = 0; j < i; j++)
            {
                double t = m->v[i][j];
                m->v[i][j] = m->v[j][i];
                m->v[j][i] = t;
            }
        }
    }

    // C++ does not support variable-length arrays
    /*
    inline mat matrix_copy(int n,  int m, double a[m][n])
    {
        mat x = matrix_new(m, n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                x->v[i][j] = a[i][j];
        return x;
    }
    */

    VISKORES_EXEC inline void matrix_copy_mat_exist(mat x, mat y)
    {
        assert(x->m==y->m);
        assert(x->n==y->n);
        for (int i = 0; i < x->m; i++)
            for (int j = 0; j < x->n; j++)
                y->v[i][j] = x->v[i][j];
    }

    VISKORES_EXEC inline mat matrix_copy_mat(mat x)
    {
        mat y = matrix_new(x->m, x->n);
        for (int i = 0; i < x->m; i++)
            for (int j = 0; j < x->n; j++)
                y->v[i][j] = x->v[i][j];
        return y;
    }

    VISKORES_EXEC inline void matrix_mul_toz(mat x, mat y, mat z)
    {
        if (x->n != y->m)
        {
            printf("error for matrix_mul_toz for dims");
            return;
        }

        for (int i = 0; i < x->m; i++)
            for (int j = 0; j < y->n; j++)
            {
                // init specific position of z as zero
                z->v[i][j] = 0;
                for (int k = 0; k < x->n; k++)
                    z->v[i][j] += x->v[i][k] * y->v[k][j];
            }
    }

    VISKORES_EXEC inline mat matrix_mul(mat x, mat y)
    {
        if (x->n != y->m)
            return 0;
        mat r = matrix_new(x->m, y->n);
        for (int i = 0; i < x->m; i++)
            for (int j = 0; j < y->n; j++)
                for (int k = 0; k < x->n; k++)
                    r->v[i][j] += x->v[i][k] * y->v[k][j];
        return r;
    }

    // compute A*U+M, the M will be added to each column of the matrix
    VISKORES_EXEC inline mat matrix_mul_add_vector(mat A, mat U, vec M)
    {
        mat AU = matrix_mul(A, U);

        assert(AU->m == M->len);

        // go through each col of the matrix
        for (int j = 0; j < AU->n; j++)
        {
            // adding vector into the matrix
            for (int i = 0; i < AU->m; i++)
            {
                AU->v[i][j] += M->v[i];
            }
        }

        return AU;
    }

    VISKORES_EXEC inline mat matrix_minor(mat x, int d)
    {
        mat m = matrix_new(x->m, x->n);
        for (int i = 0; i < d; i++)
            m->v[i][i] = 1;
        for (int i = d; i < x->m; i++)
            for (int j = d; j < x->n; j++)
                m->v[i][j] = x->v[i][j];
        return m;
    }

    /* c = a + b * s */
    VISKORES_EXEC inline double *vmadd(double a[], double b[], double s, double c[], int n)
    {
        for (int i = 0; i < n; i++)
            c[i] = a[i] + s * b[i];
        return c;
    }

    /* m = I - 2* v v^T */
    VISKORES_EXEC inline mat vmul(double v[], int n)
    {
        mat x = matrix_new(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                x->v[i][j] = -2 * v[i] * v[j];
        for (int i = 0; i < n; i++)
            x->v[i][i] += 1;

        return x;
    }

    /* ||x|| */
    VISKORES_EXEC inline double vnorm(double x[], int n)
    {
        double sum = 0;
        for (int i = 0; i < n; i++)
            sum += x[i] * x[i];
        return sqrt(sum);
    }

    VISKORES_EXEC inline bool vec_equal(vec v1, vec v2)
    {
        if (v1->len != v2->len)
        {
            return false;
        }

        for (int i = 0; i < v1->len; i++)
        {
            if (fabs(v1->v[i] - v2->v[i]) > 0.0001)
            {
                return false;
            }
        }
        return true;
    }

    VISKORES_EXEC inline void vnorm_self(vec v)
    {
        double sum = 0;
        for (int i = 0; i < v->len; i++)
            sum += v->v[i] * v->v[i];

        assert(sum != 0);

        for (int i = 0; i < v->len; i++)
        {
            v->v[i] = v->v[i] / sqrt(sum);
        }
    }

    /* y = x / d */
    // in the qr decomposition example
    // the d might be zero, which is not good, how to handle this
    VISKORES_EXEC inline double *vdiv(double x[], double d, double y[], int n)
    {
        // check the norm value
        // do not supposed to be zero
        // printf("value of d %f\n",d);
        // assert(fabs(d - 0.0) > 0.00001);
        for (int i = 0; i < n; i++)
        {
            // just set it to 0 if the norm is zero
            if (fabs(d - 0.0) < 0.00001)
            {
                y[i] = 0;
            }
            else
            {
                y[i] = x[i] / d;
            }
        }

        return y;
    }

    /* matric times a vector, result is a vector*/
    VISKORES_EXEC inline void matrix_mul_vec(mat x, vec v, vec xv)
    {
        assert(x->n == v->len);
        assert(x->m == xv->len);
        for (int i = 0; i < x->m; i++)
        {
            for (int j = 0; j < x->n; j++)
            {
                // for each row
                xv->v[i] += x->v[i][j] * v->v[j];
            }
        }
        return;
    }
    /* take c-th column of m, put in v */
    VISKORES_EXEC inline double *mcol(mat m, double *v, int c)
    {
        for (int i = 0; i < m->m; i++)
            v[i] = m->v[i][c];
        return v;
    }

    VISKORES_EXEC inline void matrix_show(mat m)
    {
        for (int i = 0; i < m->m; i++)
        {
            for (int j = 0; j < m->n; j++)
            {
                printf(" %8.3f", m->v[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    VISKORES_EXEC inline void householder(mat m, mat *R, mat *Q)
    {
        // cuda compiler does not allow this dynamic allocation
        // mat q[m->m];
        mat *q = matrix_new_array_entry(m->m, m->m, (m->m)-1);

        mat z = m, z1;
        for (int k = 0; k < m->n && k < m->m - 1; k++)
        {
            // in c++ we can not declare array by this way
            // double e[m->m], x[m->m], a;
            double a;
            vec e = vec_new(m->m);
            vec x = vec_new(m->m);


            z1 = matrix_minor(z, k);


            if (z != m)
                matrix_delete(z);
            z = z1;


            mcol(z, x->v, k);
            a = vnorm(x->v, m->m);


            if (m->v[k][k] > 0)
                a = -a;

            for (int i = 0; i < m->m; i++)
                e->v[i] = (i == k) ? 1 : 0;


            // vmadd(x, e, a, e, m->m);
            vmadd(x->v, e->v, a, e->v, m->m);
            vdiv(e->v, vnorm(e->v, m->m), e->v, m->m);
            q[k] = vmul(e->v, m->m);
            z1 = matrix_mul(q[k], z);
            if (z != m)
                matrix_delete(z);
            z = z1;

            vec_delete(e);
            vec_delete(x);

        }
        matrix_delete(z);
        *Q = q[0];
        *R = matrix_mul(q[0], m);

        for (int i = 1; i < m->n && i < m->m - 1; i++)
        {
            z1 = matrix_mul(q[i], *Q);
            if (i > 1)
                matrix_delete(*Q);
            *Q = z1;
            matrix_delete(q[i]);
        }

        
        matrix_delete(q[0]);
        //free entry itsself
        free(q);

        z = matrix_mul(*Q, m);
        matrix_delete(*R);
        *R = z;
        matrix_transpose(*Q);
    }

    VISKORES_EXEC inline bool matrix_is_upper_triangular(mat m, double tol)
    {
        // For now, only treat square matricies. */
        assert(m->m == m->n);
        for (int i = 0; i < m->m; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (fabs(m->v[i][j]) > tol)
                {
                    return false;
                }
            }
        }
        return true;
    }

    // only tested for n*n matrix and it is symetric
    VISKORES_EXEC inline void eigen_solve_eigenvalues(mat x, double tol, int max_iter, double *eigen_array)
    {
        // refer to https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition
        mat ak = matrix_copy_mat(x);


        mat qq = matrix_new_eye(x->m, x->m);

        int i = 0;
        while (true)
        {

            // it is not ok to init the empty pointer on cuda device by this way
            mat R, Q;
            // the hoiseholder will assign new r and q each time
            // TODO update householder to reuse the matrix
            householder(ak, &R, &Q);

            // mat m = matrix_mul(R, Q);
            // puts("eigen m");
            // matrix_show(m);

            matrix_mul_toz(R, Q, ak);
            // ak = matrix_copy_mat(m);
            // puts("eigen ak");
            // matrix_show(ak);
            //printf("eigen Q\n");
            //matrix_show(Q);
            //printf("eigen R\n");
            //matrix_show(R);
            // puts("m");
            // matrix_show(m);

            mat newq = matrix_mul(qq, Q);
            // update the qq to newq
            matrix_copy_mat_exist(newq, qq);

            matrix_delete(R);
            matrix_delete(Q);
            matrix_delete(newq);
            i++;
            if (matrix_is_upper_triangular(ak, tol) || i > max_iter)
            {
                // matrix_show(m);
                // printf("iter %d\n", i);
                break;
            }
            // delete m when it is not qualified
            // matrix_delete(m);
        }

        // TODO, only consider n*n matrix now
        for (int i = 0; i < ak->m; i++)
        {
            if (fabs(ak->v[i][i] - 0) < 0.00001)
            {
                eigen_array[i] = 0.0;
            }
            else
            {
                eigen_array[i] = ak->v[i][i];
            }
        }

        matrix_delete(ak);
        matrix_delete(qq);

    }

    // inverse 4*4 matrix
    // for small matrix, we use the direact inverse to solve the linear equation
    // for large matrix, we need to use other method such as qr things to solve
    // the linear system
    // there is a mesa version online

    // TODO, checking singularity for inverting the matrix

    VISKORES_EXEC inline bool invert4by4matrix(mat m, mat inv_m)
    {

        inv_m->v[0][0] = m->v[1][1] * m->v[2][2] * m->v[3][3] -
                         m->v[1][1] * m->v[2][3] * m->v[3][2] -
                         m->v[2][1] * m->v[1][2] * m->v[3][3] +
                         m->v[2][1] * m->v[1][3] * m->v[3][2] +
                         m->v[3][1] * m->v[1][2] * m->v[2][3] -
                         m->v[3][1] * m->v[1][3] * m->v[2][2];

        inv_m->v[1][0] = -m->v[1][0] * m->v[2][2] * m->v[3][3] +
                         m->v[1][0] * m->v[2][3] * m->v[3][2] +
                         m->v[2][0] * m->v[1][2] * m->v[3][3] -
                         m->v[2][0] * m->v[1][3] * m->v[3][2] -
                         m->v[3][0] * m->v[1][2] * m->v[2][3] +
                         m->v[3][0] * m->v[1][3] * m->v[2][2];

        inv_m->v[2][0] = m->v[1][0] * m->v[2][1] * m->v[3][3] -
                         m->v[1][0] * m->v[2][3] * m->v[3][1] -
                         m->v[2][0] * m->v[1][1] * m->v[3][3] +
                         m->v[2][0] * m->v[1][3] * m->v[3][1] +
                         m->v[3][0] * m->v[1][1] * m->v[2][3] -
                         m->v[3][0] * m->v[1][3] * m->v[2][1];

        inv_m->v[3][0] = -m->v[1][0] * m->v[2][1] * m->v[3][2] +
                         m->v[1][0] * m->v[2][2] * m->v[3][1] +
                         m->v[2][0] * m->v[1][1] * m->v[3][2] -
                         m->v[2][0] * m->v[1][2] * m->v[3][1] -
                         m->v[3][0] * m->v[1][1] * m->v[2][2] +
                         m->v[3][0] * m->v[1][2] * m->v[2][1];

        inv_m->v[0][1] = -m->v[0][1] * m->v[2][2] * m->v[3][3] +
                         m->v[0][1] * m->v[2][3] * m->v[3][2] +
                         m->v[2][1] * m->v[0][2] * m->v[3][3] -
                         m->v[2][1] * m->v[0][3] * m->v[3][2] -
                         m->v[3][1] * m->v[0][2] * m->v[2][3] +
                         m->v[3][1] * m->v[0][3] * m->v[2][2];

        inv_m->v[1][1] = m->v[0][0] * m->v[2][2] * m->v[3][3] -
                         m->v[0][0] * m->v[2][3] * m->v[3][2] -
                         m->v[2][0] * m->v[0][2] * m->v[3][3] +
                         m->v[2][0] * m->v[0][3] * m->v[3][2] +
                         m->v[3][0] * m->v[0][2] * m->v[2][3] -
                         m->v[3][0] * m->v[0][3] * m->v[2][2];

        inv_m->v[2][1] = -m->v[0][0] * m->v[2][1] * m->v[3][3] +
                         m->v[0][0] * m->v[2][3] * m->v[3][1] +
                         m->v[2][0] * m->v[0][1] * m->v[3][3] -
                         m->v[2][0] * m->v[0][3] * m->v[3][1] -
                         m->v[3][0] * m->v[0][1] * m->v[2][3] +
                         m->v[3][0] * m->v[0][3] * m->v[2][1];

        inv_m->v[3][1] = m->v[0][0] * m->v[2][1] * m->v[3][2] -
                         m->v[0][0] * m->v[2][2] * m->v[3][1] -
                         m->v[2][0] * m->v[0][1] * m->v[3][2] +
                         m->v[2][0] * m->v[0][2] * m->v[3][1] +
                         m->v[3][0] * m->v[0][1] * m->v[2][2] -
                         m->v[3][0] * m->v[0][2] * m->v[2][1];

        inv_m->v[0][2] = m->v[0][1] * m->v[1][2] * m->v[3][3] -
                         m->v[0][1] * m->v[1][3] * m->v[3][2] -
                         m->v[1][1] * m->v[0][2] * m->v[3][3] +
                         m->v[1][1] * m->v[0][3] * m->v[3][2] +
                         m->v[3][1] * m->v[0][2] * m->v[1][3] -
                         m->v[3][1] * m->v[0][3] * m->v[1][2];

        inv_m->v[1][2] = -m->v[0][0] * m->v[1][2] * m->v[3][3] +
                         m->v[0][0] * m->v[1][3] * m->v[3][2] +
                         m->v[1][0] * m->v[0][2] * m->v[3][3] -
                         m->v[1][0] * m->v[0][3] * m->v[3][2] -
                         m->v[3][0] * m->v[0][2] * m->v[1][3] +
                         m->v[3][0] * m->v[0][3] * m->v[1][2];

        inv_m->v[2][2] = m->v[0][0] * m->v[1][1] * m->v[3][3] -
                         m->v[0][0] * m->v[1][3] * m->v[3][1] -
                         m->v[1][0] * m->v[0][1] * m->v[3][3] +
                         m->v[1][0] * m->v[0][3] * m->v[3][1] +
                         m->v[3][0] * m->v[0][1] * m->v[1][3] -
                         m->v[3][0] * m->v[0][3] * m->v[1][1];

        inv_m->v[3][2] = -m->v[0][0] * m->v[1][1] * m->v[3][2] +
                         m->v[0][0] * m->v[1][2] * m->v[3][1] +
                         m->v[1][0] * m->v[0][1] * m->v[3][2] -
                         m->v[1][0] * m->v[0][2] * m->v[3][1] -
                         m->v[3][0] * m->v[0][1] * m->v[1][2] +
                         m->v[3][0] * m->v[0][2] * m->v[1][1];

        inv_m->v[0][3] = -m->v[0][1] * m->v[1][2] * m->v[2][3] +
                         m->v[0][1] * m->v[1][3] * m->v[2][2] +
                         m->v[1][1] * m->v[0][2] * m->v[2][3] -
                         m->v[1][1] * m->v[0][3] * m->v[2][2] -
                         m->v[2][1] * m->v[0][2] * m->v[1][3] +
                         m->v[2][1] * m->v[0][3] * m->v[1][2];

        inv_m->v[1][3] = m->v[0][0] * m->v[1][2] * m->v[2][3] -
                         m->v[0][0] * m->v[1][3] * m->v[2][2] -
                         m->v[1][0] * m->v[0][2] * m->v[2][3] +
                         m->v[1][0] * m->v[0][3] * m->v[2][2] +
                         m->v[2][0] * m->v[0][2] * m->v[1][3] -
                         m->v[2][0] * m->v[0][3] * m->v[1][2];

        inv_m->v[2][3] = -m->v[0][0] * m->v[1][1] * m->v[2][3] +
                         m->v[0][0] * m->v[1][3] * m->v[2][1] +
                         m->v[1][0] * m->v[0][1] * m->v[2][3] -
                         m->v[1][0] * m->v[0][3] * m->v[2][1] -
                         m->v[2][0] * m->v[0][1] * m->v[1][3] +
                         m->v[2][0] * m->v[0][3] * m->v[1][1];

        inv_m->v[3][3] = m->v[0][0] * m->v[1][1] * m->v[2][2] -
                         m->v[0][0] * m->v[1][2] * m->v[2][1] -
                         m->v[1][0] * m->v[0][1] * m->v[2][2] +
                         m->v[1][0] * m->v[0][2] * m->v[2][1] +
                         m->v[2][0] * m->v[0][1] * m->v[1][2] -
                         m->v[2][0] * m->v[0][2] * m->v[1][1];

        double det = m->v[0][0] * inv_m->v[0][0] +
                     m->v[0][1] * inv_m->v[1][0] +
                     m->v[0][2] * inv_m->v[2][0] +
                     m->v[0][3] * inv_m->v[3][0];

        if (det == 0){
            printf("singular matrix\n");
            //matrix_show(m);
            return false;

        }
            

        det = 1.0 / det;

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                inv_m->v[i][j] = inv_m->v[i][j] * det;
            }
        }

        return true;
    }

    // go through each eigen values
    // if the eigen value is 0, the associated eigen vector is 0
    // other wise, using the inverse iteration algorithm to compute the eigen vector
    // https://en.wikipedia.org/wiki/Inverse_iteration
    VISKORES_EXEC inline mat eigen_solve_eigen_vectors(mat m, double *eigen_value_array, int len_eigen_vec, int num_eigen_value, int iter_max)
    {

        // only works for 4*4 now, since the matirc reverse is designed for 4*4
        // add more flexible linear system solver in future
        assert(m->m == 4);
        assert(m->n == 4);

        // create the empty eigen vectors matrix
        // raw of matrix represents the size of eigen vec
        // col of matrix represents the number of eigen values
        mat e_vec = matrix_new(len_eigen_vec, num_eigen_value);

        // reuse this matrix each time
        mat m_minus_lambda_i = matrix_new(m->m, m->n);
        mat m_minus_lambda_i_inv = matrix_new(m->m, m->n);

        // init b0
        vec b_curr = vec_new(len_eigen_vec);
        vec b_prev = vec_new(len_eigen_vec);
        // puts("init bcurr");
        // vec_show(&b_curr);

        // init as 1
        for (int i = 0; i < len_eigen_vec; i++)
        {
            b_prev->v[i] = 1.0;
        }
        // puts("init b_prev");
        // vec_show(&b_prev);
        //  go through each eigen value j
        //  this is the colm of matrix
        for (int j = 0; j < num_eigen_value; j++)
        {
            if (fabs(eigen_value_array[j] - 0) < 0.0001)
            {
                continue;
            }
            // finding iegen vector in an iterative way
            int iter_num = 0;
            // Preturb the eigenvalue a litle to prevent our right hand side matrix
            // from becoming singular.
            // double lambda = eigen_value_array[j] + ((double)rand() / (double)RAND_MAX) * 0.000001;
            double lambda = eigen_value_array[j]+0.000001;
            //  reset the m_minus_lambda_i
            for (int ii = 0; ii < m->m; ii++)
            {
                for (int jj = 0; jj < m->n; jj++)
                {
                    if (ii == jj)
                    {
                        m_minus_lambda_i->v[ii][jj] = m->v[ii][jj] - lambda;
                    }
                    else
                    {
                        m_minus_lambda_i->v[ii][jj] = m->v[ii][jj];
                    }
                }
            }

            // puts("m_minus_lambda_i");
            // matrix_show(m_minus_lambda_i);
            //  solve equation bk+1 = m_minus_lambda_i_rev * bk i
            bool invertok = invert4by4matrix(m_minus_lambda_i, m_minus_lambda_i_inv);
            assert(invertok == true);
            // puts("m_minus_lambda_i_inv");
            // matrix_show(m_minus_lambda_i_inv);

            while (iter_num < iter_max)
            {
                // check the diff between curr and prev
                matrix_mul_vec(m_minus_lambda_i_inv, b_prev, b_curr);
                // vec_show(&b_curr);
                //  norm
                vnorm_self(b_curr);

                if (vec_equal(b_curr, b_prev))
                {
                    // reult convert
                    break;
                }

                // assign curr to prev
                for (int i = 0; i < b_curr->len; i++)
                {
                    b_prev->v[i] = b_curr->v[i];
                }

                iter_num++;
            }
            // assign to the matrix
            // printf("iter number %d\n", iter_num);
            for (int i = 0; i < len_eigen_vec; i++)
            {
                e_vec->v[i][j] = b_curr->v[i];
            }
        }

        matrix_delete(m_minus_lambda_i);
        matrix_delete(m_minus_lambda_i_inv);

        vec_delete(b_curr);
        vec_delete(b_prev);

        // if it is zero, the ith column of the eigen vector is zero

        return e_vec;
    }

    // input m and get its eigen vector decomposition a where a*a^t = m
    VISKORES_EXEC inline mat eigen_vector_decomposition(mat x)
    {

        // assuming m has eigen value and eigen vectors
        // assuming it is not singular matrix

        double result[4];
        eigen_solve_eigenvalues(x, 0.0001, 20, result);

        // update this when we have flexible linear system solver
        assert(x->m == 4);
        assert(x->n == 4);
        mat eigen_vectors = eigen_solve_eigen_vectors(x, result, 4, 4, 20);

        // create the diaganal matrix
        mat diag = matrix_new(x->m, x->n);

        for (int i = 0; i < x->m; i++)
        {
            diag->v[i][i] = sqrt(result[i]);
        }

        mat A = matrix_mul(eigen_vectors, diag);

        matrix_delete(eigen_vectors);
        matrix_delete(diag);

        return A;
    }

    // gaussian distribution sampling matrix
    VISKORES_EXEC inline mat norm_sampling(int row, int samples)
    {
        // the row is legth of each sample vector
        // the col is the number of samples

        // use the thrust library on gpu here

#ifdef VISKORES_CUDA
        thrust::minstd_rand rng;
        thrust::random::normal_distribution<double> norm;
#else
        std::mt19937 rng;
        rng.seed(std::mt19937::default_seed);
        std::normal_distribution<double> norm;
#endif // VISKORES_CUDA

        mat sample_matrix = matrix_new(row, samples);
        for (int j = 0; j < samples; j++)
        {
            for (int i = 0; i < row; i++)
            {
                // using other sample mechanism such as thrust as needed
                sample_matrix->v[i][j] = norm(rng);
            }
        }

        return sample_matrix;
    }
}

#endif
