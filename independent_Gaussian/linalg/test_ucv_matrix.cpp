
#include "./ucv_matrix.h"
#include <assert.h>

using namespace UCVMATH;

// for symetric matrix
double in4_1[4][4] = {
    {0.20, 0.60, 0.40, 0.80},
    {0.60, 1.80, 1.20, 2.40},
    {0.40, 1.20, 0.80, 1.60},
    {0.80, 2.40, 1.60, 3.20}};

double in3_1[3][3] = {
    {1.0, 3.0, 7.0},
    {3.0, 2.0, 6.0},
    {7.0, 6.0, 5.0}};

double in4_2[4][4] = {
    {1.0, 3.0, 7.0, 8.0},
    {3.0, 2.0, 6.0, 7.0},
    {7.0, 6.0, 5.0, 6.0},
    {8.0, 7.0, 6.0, 5.0}};

double in4_3[4][4] = {
    {0.0, 0.0, 0.0, 0.0},
    {0.0, 20.0, 60.0, 40.0},
    {0.0, 60.0, 180.0, 120.0},
    {0.0, 40.0, 120.0, 80.0}};

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

void eigen_values_3by3()
{
    // mat x = matrix_copy(3, in2, 3);
    // mat x = matrix_copy(4, in1, 4);
    // mat x = matrix_copy(4, in3, 4);
    printf("---eigen_values_3by3\n");

    mat x = matrix_new(3, 3);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            x->v[i][j] = in3_1[i][j];
        }
    }

    double result[3]={0};
    eigen_solve_eigenvalues(x, 0.0001, 5, result);

    for (int i = 0; i < 3; i++)
    {
        printf("%f ", result[i]);
    }

    printf("\n");
    assert(equal_double(result[0], 13.94740) == 1);
    assert(equal_double(result[1], -4.67428) == 1);
    assert(equal_double(result[2], -1.27311) == 1);

    matrix_delete(x);
}

void eigen_values_4by4()
{
    printf("--eigen_values_4by4\n");

    int dim = 4;
    mat x = matrix_new(dim, dim);
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x->v[i][j] = in4_1[i][j];
        }
    }

    double result[4]={0};
    eigen_solve_eigenvalues(x, 0.0001, 20, result);

    for (int i = 0; i < dim; i++)
    {
        printf("%f ", result[i]);
    }

    printf("\n");
    // the assert only works for debug case
    assert(equal_double(result[0], 6.0) == 1);
    assert(equal_double(result[1], 0.0) == 1);
    assert(equal_double(result[2], 0.0) == 1);
    assert(equal_double(result[3], 0.0) == 1);

    matrix_delete(x);
}

void basic_qr_test()
{
    printf("---basic_qr_test\n");
    mat R, Q;
    int msize = 4;
    mat x = matrix_new(msize, msize);

    for (int i = 0; i < msize; i++)
    {
        for (int j = 0; j < msize; j++)
        {
            x->v[i][j] = in4_1[i][j];
        }
    }

    puts("original matric");
    matrix_show(x);

    householder(x, &R, &Q);

    puts("Q");
    matrix_show(Q);
    puts("R");
    matrix_show(R);

    // to show their product is the input matrix
    mat m = matrix_mul(Q, R);
    puts("Q * R");
    matrix_show(m);

    matrix_delete(x);
    matrix_delete(R);
    matrix_delete(Q);
    matrix_delete(m);
}

void test_invert_4by4matrix()
{
    int dim = 4;
    mat x = matrix_new(dim, dim);
    mat x_inv = matrix_new(dim, dim);
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x->v[i][j] = in4_2[i][j];
        }
    }

    bool inv_ok = invert4by4matrix(x, x_inv);

    puts("x");
    matrix_show(x);
    puts("x_inv");
    matrix_show(x_inv);

    mat c1 = matrix_mul(x, x_inv);

    puts("x*x_inv");
    matrix_show(c1);

    mat c2 = matrix_mul(x_inv, x);

    puts("x_inv*x");
    matrix_show(c2);

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (i == j)
            {
                assert(equal_double(c1->v[i][j], 1.0) == 1);
                assert(equal_double(c1->v[i][j], 1.0) == 1);
            }
            else
            {
                assert(equal_double(c2->v[i][j], 0.0) == 1);
                assert(equal_double(c2->v[i][j], 0.0) == 1);
            }
        }
    }

    matrix_delete(x);
    matrix_delete(x_inv);
    matrix_delete(c1);
    matrix_delete(c2);

    assert(inv_ok == true);
}

void eigen_vectors_4by4()
{

    printf("---test eigen_vectors_4by4\n");
    int dim = 4;
    mat x = matrix_new(dim, dim);
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x->v[i][j] = in4_3[i][j];
        }
    }

    matrix_show(x);

    mat R, Q;
    householder(x, &R, &Q);

    puts("Q");
    matrix_show(Q);
    puts("R");
    matrix_show(R);

    mat m = matrix_mul(Q, R);
    puts("Q * R");
    matrix_show(m);

    // the assert only works for debug case

    double result[4]={0};
    eigen_solve_eigenvalues(x, 0.0001, 20, result);
    for (int i = 0; i < 4; i++)
    {
        printf(" %f", result[i]);
    }
    printf("\n");
    assert(equal_double(result[0], 0) == 1);
    assert(equal_double(result[1], 280.0) == 1);
    assert(equal_double(result[2], 0.0) == 1);
    assert(equal_double(result[3], 0.0) == 1);

    mat eigen_vectors = eigen_solve_eigen_vectors(x, result, 4, 4, 20);
    matrix_show(eigen_vectors);
    assert(equal_double(eigen_vectors->v[0][1], 0.0) == 1);
    assert(equal_double(eigen_vectors->v[1][1], 0.2672) == 1);
    assert(equal_double(eigen_vectors->v[2][1], 0.8017) == 1);
    assert(equal_double(eigen_vectors->v[3][1], 0.5345) == 1);

    matrix_delete(Q);
    matrix_delete(R);
    matrix_delete(m);
    matrix_delete(x);
    matrix_delete(eigen_vectors);
}

void test_eigen_vectors_decomposition()
{
    printf("---test_eigen_vectors_decomposition\n");

    int dim = 4;
    mat x = matrix_new(dim, dim);
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x->v[i][j] = in4_3[i][j];
        }
    }

    mat A = eigen_vector_decomposition(x);

    mat A_trans = matrix_copy_mat(A);
    matrix_transpose(A_trans);

    mat rst = matrix_mul(A, A_trans);

    puts("A");
    matrix_show(A);
    puts("rst");
    matrix_show(rst);

    assert(equal_matrix(rst, x) == 1);

    matrix_delete(rst);
    matrix_delete(A);
    matrix_delete(A_trans);
    matrix_delete(x);
}

int main()
{
    basic_qr_test();
    eigen_values_3by3();
    eigen_values_4by4();
    test_invert_4by4matrix();
    eigen_vectors_4by4();
    test_eigen_vectors_decomposition();
    return 0;
}