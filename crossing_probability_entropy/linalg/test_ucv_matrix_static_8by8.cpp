
#include "./ucv_matrix_static_8by8.h"
#include <assert.h>

using namespace UCVMATH;

// for symetric matrix

double mat_0[8][8] = {
    {1, 1, 0, 1, 0, 1, 0, 1},
    {1, 2, 1, 0, 1, 0, 1, 0},
    {0, 1, 3, 1, 0, 1, 0, 1},
    {1, 0, 1, 4, 1, 0, 1, 0},
    {0, 1, 0, 1, 5, 1, 0, 1},
    {1, 0, 1, 0, 1, 6, 1, 0},
    {0, 1, 0, 1, 0, 1, 7, 1},
    {1, 0, 1, 0, 1, 0, 1, 8}};

int equal_double(double a, double b)
{
    if (fabs(a - b) < 0.0001)
    {
        return 1;
    }
    return 0;
}

int equal_matrix(mat a, mat b)
{
    if (a->m != b->m)
    {
        return false;
    }
    if (a->n != b->n)
    {
        return false;
    }
    for (int i = 0; i < a->m; i++)
    {
        for (int j = 0; j < a->n; j++)
        {
            if (equal_double(a->v[i][j], b->v[i][j]) == 0)
            {
                return false;
            }
        }
    }
    return true;
}

void test_eigen_values_8by8()
{
    printf("--test_eigen_values_8by8\n");

    mat_t x;
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            x.v[i][j] = mat_0[i][j];
        }
    }

    double result[8] = {0};
    eigen_solve_eigenvalues(&x, 0.0001, 20, result);

    for (int i = 0; i < 8; i++)
    {
        printf("%f ", result[i]);
    }

    printf("\n");

    // the assert only works for debug case
    assert(equal_double(result[0], 9.6789) == 1);
    assert(equal_double(result[1], 7.1721) == 1);
    assert(equal_double(result[2], 6.1691) == 1);
    assert(equal_double(result[3], 4.9999) == 1);
    assert(equal_double(result[4], 4.0001) == 1);
    assert(equal_double(result[5], 2.8321) == 1);
    assert(equal_double(result[6], 1.8265) == 1);
    assert(equal_double(result[7], -0.6789) == 1);
}

void test_basic_qr()
{
    printf("---test_basic_qr\n");
    mat_t R, Q;
    int msize = 8;
    mat_t x;

    for (int i = 0; i < msize; i++)
    {
        for (int j = 0; j < msize; j++)
        {
            x.v[i][j] = mat_0[i][j];
        }
    }

    puts("original matrix");
    matrix_show(&x);

    householder(&x, &R, &Q);

    puts("Q");
    matrix_show(&Q);
    puts("R");
    matrix_show(&R);

    // check orthoganal
    bool ifupper = matrix_is_upper_triangular(&R, 0.00001);
    assert(ifupper == true);

    // to show their product is the input matrix
    mat_t m = matrix_mul(&Q, &R);
    puts("Q * R");
    matrix_show(&m);

    assert(equal_matrix(&m, &x) == 1);
}

void test_invert_8by8matrix()
{
    int dim = 8;
    mat_t x;
    mat_t x_inv;
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x.v[i][j] = mat_0[i][j];
        }
    }

    invert8by8matrix(&x, &x_inv);

    puts("x");
    matrix_show(&x);
    puts("x_inv");
    matrix_show(&x_inv);

    mat_t c1 = matrix_mul(&x, &x_inv);

    puts("x*x_inv");
    matrix_show(&c1);

    mat_t c2 = matrix_mul(&x_inv, &x);

    puts("x_inv*x");
    matrix_show(&c2);

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            if (i == j)
            {
                assert(equal_double(c1.v[i][j], 1.0) == 1);
                assert(equal_double(c1.v[i][j], 1.0) == 1);
            }
            else
            {
                assert(equal_double(c2.v[i][j], 0.0) == 1);
                assert(equal_double(c2.v[i][j], 0.0) == 1);
            }
        }
    }
}

void test_eigen_vectors_8by8()
{

    printf("---test test_eigen_vectors_8by8\n");
    constexpr int dim = 8;
    mat_t x;
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x.v[i][j] = mat_0[i][j];
        }
    }

    matrix_show(&x);

    mat_t R, Q;
    householder(&x, &R, &Q);

    puts("Q");
    matrix_show(&Q);
    puts("R");
    matrix_show(&R);

    mat_t m = matrix_mul(&Q, &R);
    puts("Q * R");
    matrix_show(&m);

    // the assert only works for debug case

    double result[dim] = {0};
    eigen_solve_eigenvalues(&x, 0.0001, 20, result);
    for (int i = 0; i < dim; i++)
    {
        printf(" %f", result[i]);
    }
    printf("\n");

    mat_t eigen_vectors = eigen_solve_eigen_vectors(&x, result, dim, dim, 20);
    matrix_show(&eigen_vectors);

    // do the checking to see if
    // A*vx=lambda*vx

    for (int j = 0; j < DIM; j++)
    {
        vec_t avx;
        vec_t eigenvct;
        double eigenvalue = result[j];

        // extract ith column eigen vector list
        for (int i = 0; i < DIM; i++)
        {
            eigenvct.v[i] = eigen_vectors.v[i][j];
        }

        // vec_show(&eigenvct);

        matrix_mul_vec(&x, &eigenvct, &avx);

        // compare vector avx and lambda*eigenv
        for (int i = 0; i < DIM; i++)
        {
            eigenvct.v[i] = eigenvalue * eigenvct.v[i];
        }

        // vec_show(&avx);
        // vec_show(&eigenvct);

        for (int i = 0; i < DIM; i++)
        {
            // there are some accumulated errors
            // we have three digit precisions
            if (fabs(avx.v[i] - eigenvct.v[i]) > 0.001){
                printf("eigen value %f two number %f %f\n", eigenvalue, avx.v[i], eigenvct.v[i]);
                assert(false);
            }
        }
    }
}
void test_eigen_vectors_decomposition()
{
    printf("---test_eigen_vectors_decomposition\n");

    int dim = 8;
    mat_t x;
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x.v[i][j] = mat_0[i][j];
        }
    }

    mat_t A = eigen_vector_decomposition(&x);

    mat_t A_trans = A;
    matrix_transpose(&A_trans);

    mat_t rst = matrix_mul(&A, &A_trans);

    puts("A");
    matrix_show(&A);
    puts("rst");
    matrix_show(&rst);

    assert(equal_matrix(&rst, &x) == 1);
}

void test_basic_operations()
{

    printf("---test_basic_operations");

    mat_t a;
    vec_t u;
    vec_t m;

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            a.v[i][j] = i + 1;
        }
        u.v[i] = 1;
        m.v[i] = i + 1;
    }

    vec_t auv = matrix_mul_vec_add_vec(&a, &u, &m);
    vec_show(&auv);

    for (int i = 0; i < 8; i++)
    {
        assert(auv.v[i] == (i + 1) * 9);
    }
}

int main()
{
    test_basic_operations();
    test_basic_qr();
    test_invert_8by8matrix();
    test_eigen_values_8by8();
    test_eigen_vectors_8by8();
    //this can only work for the case where eigen value is >=0
    //test_eigen_vectors_decomposition();
    return 0;
}
