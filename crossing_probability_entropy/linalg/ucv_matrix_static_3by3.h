#ifndef UCV_MATRIX_H_STATIC_THREE
#define UCV_MATRIX_H_STATIC_THREE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <cassert>

#include <vtkm/Types.h>

#ifdef VTKM_CUDA
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#else
// using the std library
#include <random>
#endif // VTKM_CUDA

//Currently, we just use the simple idea
//namely the namespace to distinguish different matrix declaration
namespace UCVMATH_THREE
{
    // typical 3*3 matrix
    // this should be updated as needed to support matrix
    // with different size, maybe use the macro defination similar to this
    // https://stackoverflow.com/questions/34820324/macro-for-dynamic-types-in-c
    // or maybe use the code generation tool in future
     
    #define MSIZE 3

    struct mat_t
    {
        int m = MSIZE, n = MSIZE; // m is row, n is column
        double v[MSIZE][MSIZE] = {{0}};
    };
    using mat = mat_t*;

    struct vec_t
    {
        int len = MSIZE;
        double v[MSIZE] = {0};
    };
    using vec = vec_t*;

    VTKM_EXEC inline vec_t vec_new(int len)
    {
        assert(len == MSIZE);
        vec_t v;
        for (int i = 0; i < len; i++)
        {
            v.v[i] = 0;
        }
        return v;
    }

    VTKM_EXEC inline void vec_show(vec v)
    {

        for (int i = 0; i < v->len; i++)
        {
            printf("%lf ", v->v[i]);
        }
        printf("\n");
    }

    VTKM_EXEC inline mat_t matrix_new_eye(int m, int n)
    {
        mat_t x;
        assert(m == MSIZE);
        assert(n == MSIZE);
        x.m = m;
        x.n = n;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    x.v[i][j] = 1;
                }
                else
                {
                    x.v[i][j] = 0;
                }
            }
        }
        return x;
    }

    VTKM_EXEC inline void matrix_transpose(mat m)
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

    VTKM_EXEC inline void matrix_mul_toz(mat x, mat y, mat z)
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

    VTKM_EXEC inline mat_t matrix_mul(mat x, mat y)
    {
        assert(x->n == y->m);
        mat_t r;
        for (int i = 0; i < x->m; i++)
            for (int j = 0; j < y->n; j++)
                for (int k = 0; k < x->n; k++)
                    r.v[i][j] += x->v[i][k] * y->v[k][j];
        return r;
    }

    // compute A*U+M, the M will be added to each column of the matrix
    VTKM_EXEC inline vec_t matrix_mul_vec_add_vec(mat A, vec U, vec M)
    {

        assert(A->n == U->len);
        assert(A->m == M->len);

        vec_t AUM;

        for (int i = 0; i < A->m; i++)
        {
            // adding vector into the matrix
            for (int j = 0; j < A->n; j++)
            {
                AUM.v[i] += (A->v[i][j] * U->v[j]);
            }
            AUM.v[i] += M->v[i];
        }

        return AUM;
    }

    // compute A*U+M, the M will be added to each column of the matrix
    VTKM_EXEC inline mat_t matrix_mul_add_vector(mat A, mat U, vec M)
    {
        mat_t AU = matrix_mul(A, U);

        assert(AU.m == M->len);

        // go through each col of the matrix
        for (int j = 0; j < AU.n; j++)
        {
            // adding vector into the matrix
            for (int i = 0; i < AU.m; i++)
            {
                AU.v[i][j] += M->v[i];
            }
        }

        return AU;
    }

    VTKM_EXEC inline mat_t matrix_minor(mat x, int d)
    {
        mat_t m;
        for (int i = 0; i < d; i++)
            m.v[i][i] = 1;
        for (int i = d; i < x->m; i++)
            for (int j = d; j < x->n; j++)
                m.v[i][j] = x->v[i][j];
        return m;
    }

    // c = a + b * s
    VTKM_EXEC inline double *vmadd(double a[], double b[], double s, double c[], int n)
    {
        for (int i = 0; i < n; i++)
            c[i] = a[i] + s * b[i];
        return c;
    }

    // m = I - 2* v v^T
    VTKM_EXEC inline mat_t I_minus_vmul(double v[], int n)
    {
        mat_t x;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                x.v[i][j] = -2 * v[i] * v[j];
        // diagnal elements add 1
        for (int i = 0; i < n; i++)
            x.v[i][i] += 1;

        return x;
    }

    // ||x||
    VTKM_EXEC inline double vnorm(vec v)
    {
        double sum = 0;
        for (int i = 0; i < v->len; i++)
            sum += v->v[i] * v->v[i];
        return sqrt(sum);
    }

    VTKM_EXEC inline bool vec_equal(vec v1, vec v2)
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

    VTKM_EXEC inline void vnorm_self(vec v)
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

    // y = x / d
    // in the qr decomposition example
    // the d might be zero, which is not good, how to handle this
    VTKM_EXEC inline double *vdiv(double x[], double d, double y[], int n)
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

    // matric times a vector, result is a vector
    VTKM_EXEC inline void matrix_mul_vec(mat x, vec v, vec xv)
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
    // take c-th column of m, put in v
    VTKM_EXEC inline double *mcol(mat m, double *v, int c)
    {
        for (int i = 0; i < m->m; i++)
            v[i] = m->v[i][c];
        return v;
    }

    VTKM_EXEC inline void matrix_show(mat m)
    {
        for (int i = 0; i < m->m; i++)
        {
            for (int j = 0; j < m->n; j++)
            {
                printf(" %8.7f", m->v[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    // m is the original matrix
    // Q and R are matrix we want
    VTKM_EXEC inline void householder(mat m, mat R, mat Q)
    {
        // cuda compiler does not allow this dynamic allocation
        // mat q[m->m];
        // mat *q = matrix_new_array_entry(m->m, m->m, (m->m) - 1);
        // create a list, the 0 position is A the 1 position is H1...
        // the 3th position is H3
        assert(m->m == m->n);
        assert(m->m == MSIZE);

        mat_t mat_list[MSIZE];
        mat_list[0] = *m;
        *R = *m;
        for (int k = 1; k < m->m; k++)
        {
            // in c++ we can not declare array by this way
            // double e[m->m], x[m->m], a;
            double a_norm;
            vec_t v_e;
            vec_t v_a;

            // select valid part from the original H
            // start from 0
            // extract the init matrix from the previous status
            // the *R position is Hk*Hk-1*Hk-2..H1*A
            mat_t H_init = matrix_minor(R, k - 1);

            // printf("hhd iter%d\n",k);
            // puts("h init");
            // matrix_show(&H_init);
            //  select k-1th colum to v_a
            mcol(&H_init, v_a.v, (k - 1));

            // puts("a colum");
            // vec_show(&v_a);

            a_norm = vnorm(&v_a);

            // printf("a norm %f\n",a_norm);
            //  since the vector decrease each time
            //  this thing becomes the first one after decreasing
            if (m->v[k - 1][k - 1] > 0)
                a_norm = -a_norm;

            // init e vector as 1
            for (int i = 0; i < m->m; i++)
                v_e.v[i] = (i == (k - 1)) ? 1 : 0;

            // vec_a+sca(b)*vec(c)
            // reuse the e and put v_i into it
            vmadd(v_a.v, v_e.v, a_norm, v_e.v, m->m);
            // puts("v vec");
            // vec_show(&v_a);

            //(v1*v1_t)/(v1*t*v1) = (v1/v1_norm)*(v1_t/v1_norm)
            // there are sqrt in norm, two sqrt times together
            // equals to the original norm
            vdiv(v_e.v, vnorm(&v_e), v_e.v, m->m);

            // i-v*v
            // compute Hi
            mat_list[k] = I_minus_vmul(v_e.v, m->m);

            // puts("H");
            // matrix_show(&mat_list[k]);

            // update Q
            if (k == 1)
            {
                *Q = mat_list[k];
            }
            else
            {
                *Q = matrix_mul(Q, &mat_list[k]);
            }

            // update R
            *R = matrix_mul(&mat_list[k], R);
        }
    }

    VTKM_EXEC inline bool matrix_is_upper_triangular(mat m, double tol)
    {
        // For now, only treat square matricies.
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
    VTKM_EXEC inline void eigen_solve_eigenvalues(mat x, double tol, int max_iter, double *eigen_array)
    {
        // refer to https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition
        mat_t ak = *x;

        mat_t qq = matrix_new_eye(x->m, x->m);

        for (int i = 0; i < max_iter; ++i)
        {

            // it is not ok to init the empty pointer on cuda device by this way
            mat_t R, Q;
            // the hoiseholder will assign new r and q each time
            // TODO update householder to reuse the matrix
            householder(&ak, &R, &Q);

            // mat m = matrix_mul(R, Q);
            // puts("eigen m");
            // matrix_show(m);

            matrix_mul_toz(&R, &Q, &ak);
            // ak = matrix_copy_mat(m);
            // puts("eigen ak");
            // matrix_show(ak);
            // printf("eigen Q\n");
            // matrix_show(Q);
            // printf("eigen R\n");
            // matrix_show(R);
            // puts("m");
            // matrix_show(m);

            mat_t newq = matrix_mul(&qq, &Q);
            // update the qq to newq
            qq = newq;

            if (matrix_is_upper_triangular(&ak, tol))
            {
                // matrix_show(m);
                // printf("iter %d\n", i);
                break;
            }
            // delete m when it is not qualified
        }

        // TODO, only consider n*n matrix now
        for (int i = 0; i < ak.m; i++)
        {
            if (fabs(ak.v[i][i] - 0) < tol)
            {
                eigen_array[i] = 0.0;
            }
            else
            {
                eigen_array[i] = ak.v[i][i];
            }
        }
    }

    // inverse 3*3 matrix
    // for small matrix, we use the direact inverse to solve the linear equation
    // for large matrix, we need to use other method such as qr things to solve
    // the linear system
    // there is a mesa version online

    // TODO, checking singularity for inverting the matrix
    // refer to https://ardoris.wordpress.com/2008/07/18/general-formula-for-the-inverse-of-a-3x3-matrix/

    VTKM_EXEC inline bool invert3by3matrix(mat m, mat inv_m)
    {

        inv_m->v[0][0] = m->v[1][1]*m->v[2][2]-m->v[1][2]*m->v[2][1];

        inv_m->v[1][0] = m->v[1][2]*m->v[2][0]-m->v[1][0]*m->v[2][2];

        inv_m->v[2][0] = m->v[1][0]*m->v[2][1]-m->v[1][1]*m->v[2][0];

        inv_m->v[0][1] = m->v[0][2]*m->v[2][1]-m->v[0][1]*m->v[2][2];

        inv_m->v[1][1] = m->v[0][0]*m->v[2][2]-m->v[0][2]*m->v[2][0];

        inv_m->v[2][1] = m->v[0][1]*m->v[2][0]-m->v[0][0]*m->v[2][1];

        inv_m->v[0][2] = m->v[0][1]*m->v[1][2]-m->v[0][2]*m->v[1][1];

        inv_m->v[1][2] = m->v[0][2]*m->v[1][0]-m->v[0][0]*m->v[1][2];

        inv_m->v[2][2] = m->v[0][0]*m->v[1][1]-m->v[0][1]*m->v[1][0];


        double det = m->v[0][0]*(m->v[1][1]*m->v[2][2]-m->v[1][2]*m->v[2][1])
         - m->v[0][1]*(m->v[1][0]*m->v[2][2]-m->v[1][2]*m->v[2][0])
         + m->v[0][2]*(m->v[1][0]*m->v[2][1]-m->v[1][1]*m->v[2][0]); 

        if (det == 0)
        {
            printf("singular matrix\n");
            // matrix_show(m);
            return false;
        }

        det = 1.0 / det;

        for (int i = 0; i < MSIZE; i++)
        {
            for (int j = 0; j < MSIZE; j++)
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
    VTKM_EXEC inline mat_t eigen_solve_eigen_vectors(mat m, double *eigen_value_array, int len_eigen_vec, int num_eigen_value, int iter_max)
    {

        // only works for 3*3 now, since the matirc reverse is designed for 3*3
        // add more flexible linear system solver in future
        assert(m->m == MSIZE);
        assert(m->n == MSIZE);

        // create the empty eigen vectors matrix
        // raw of matrix represents the size of eigen vec
        // col of matrix represents the number of eigen values
        mat_t e_vec;

        // reuse this matrix each time
        mat_t m_minus_lambda_i;
        mat_t m_minus_lambda_i_inv;

        // init b0
        vec_t b_curr;
        vec_t b_prev;
        // puts("init bcurr");
        // vec_show(&b_curr);

        // puts("init b_prev");
        // vec_show(&b_prev);
        //  go through each eigen value j
        //  this is the colm of matrix
        for (int j = 0; j < num_eigen_value; j++)
        {
            if (fabs(eigen_value_array[j] - 0) < 0.000001)
            {
                continue;
            }
            // finding iegen vector in an iterative way
            int iter_num = 0;
            // Preturb the eigenvalue a litle to prevent our right hand side matrix
            // from becoming singular.
            // double lambda = eigen_value_array[j] + ((double)rand() / (double)RAND_MAX) * 0.000001;
            double lambda = eigen_value_array[j] + 0.00001;
            //  reset the m_minus_lambda_i
            for (int ii = 0; ii < m->m; ii++)
            {
                for (int jj = 0; jj < m->n; jj++)
                {
                    if (ii == jj)
                    {
                        m_minus_lambda_i.v[ii][jj] = m->v[ii][jj] - lambda;
                    }
                    else
                    {
                        m_minus_lambda_i.v[ii][jj] = m->v[ii][jj];
                    }
                }
            }

            // puts("m_minus_lambda_i");
            // matrix_show(m_minus_lambda_i);
            //  solve equation bk+1 = m_minus_lambda_i_rev * bk i
            bool invertok = invert3by3matrix(&m_minus_lambda_i, &m_minus_lambda_i_inv);
            assert(invertok == true);
            // puts("m_minus_lambda_i_inv");
            // matrix_show(m_minus_lambda_i_inv);

            // init as 1
            for (int i = 0; i < len_eigen_vec; i++)
            {
                b_prev.v[i] = 1.0;
            }

            while (iter_num < iter_max)
            {
                // check the diff between curr and prev
                matrix_mul_vec(&m_minus_lambda_i_inv, &b_prev, &b_curr);
                // vec_show(&b_curr);
                //  norm
                vnorm_self(&b_curr);

                if (vec_equal(&b_curr, &b_prev))
                {
                    // reult convert
                    break;
                }

                // assign curr to prev
                for (int i = 0; i < b_curr.len; i++)
                {
                    b_prev.v[i] = b_curr.v[i];
                }

                iter_num++;
            }
            // assign to the matrix
            // printf("iter number %d\n", iter_num);
            for (int i = 0; i < len_eigen_vec; i++)
            {
                e_vec.v[i][j] = b_curr.v[i];
            }
        }

        // if it is zero, the ith column of the eigen vector is zero

        return e_vec;
    }

    // input m and get its eigen vector decomposition a where a*a^t = m
    VTKM_EXEC inline mat_t eigen_vector_decomposition(mat x)
    {

        // assuming m has eigen value and eigen vectors
        // assuming it is not singular matrix

        double result[MSIZE];
        eigen_solve_eigenvalues(x, 0.000001, 30, result);

        // update this when we have flexible linear system solver
        assert(x->m == MSIZE);
        assert(x->n == MSIZE);
        mat_t eigen_vectors = eigen_solve_eigen_vectors(x, result, MSIZE, MSIZE, 30);

        // create the diaganal matrix
        mat_t diag;

        for (int i = 0; i < x->m; i++)
        {
            diag.v[i][i] = sqrt(result[i]);
        }

        mat_t A = matrix_mul(&eigen_vectors, &diag);

        return A;
    }

    VTKM_EXEC inline vec_t norm_sampling_vec(int row)
    {
        assert(row == MSIZE);
#ifdef VTKM_CUDA
        thrust::minstd_rand rng;
        thrust::random::normal_distribution<double> norm;
#else
        std::mt19937 rng;
        rng.seed(std::mt19937::default_seed);
        std::normal_distribution<double> norm;
#endif // VTKM_CUDA
        vec_t samplev;
        for (int i = 0; i < row; i++)
        {
            // using other sample mechanism such as thrust as needed
            samplev.v[i] = norm(rng);
        }
        return samplev;
    }

}

#endif
