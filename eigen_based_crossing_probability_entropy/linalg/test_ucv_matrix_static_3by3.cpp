
#include "./ucv_matrix_static_3by3.h"
#include <assert.h>

using namespace UCVMATH_THREE;

// for symetric matrix

double in_0[3][3] = {
    {1,-1,4},
    {1,4,-2},
    {1,4,2}};

double in_1[3][3] = {
    {0.20, 0.60, 0.40},
    {0.60, 1.80, 1.20},
    {0.40, 1.20, 0.80}};

double in_2[3][3] = {
    {1.0, 3.0, 7.0},
    {3.0, 2.0, 6.0},
    {7.0, 6.0, 5.0}};

double in_3[3][3] = {
    {0.0, 0.0, 0.0},
    {0.0, 20.0, 60.0},
    {0.0, 60.0, 180.0}};

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
    printf("--eigen_values_3by3\n");

    mat_t x ;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            x.v[i][j] = in_1[i][j];
        }
    }

    double result[3]={0};
    eigen_solve_eigenvalues(&x, 0.0001, 20, result);

    for (int i = 0; i < 3; i++)
    {
        printf("%f ", result[i]);
    }

    printf("\n");
    // the assert only works for debug case
    assert(equal_double(result[0], 2.8) == 1);
    assert(equal_double(result[1], 0.0) == 1);
    assert(equal_double(result[2], 0.0) == 1);

}

void basic_qr_test()
{
    printf("---basic_qr_test\n");
    mat_t R, Q;
    int msize = 3;
    mat_t x;

    for (int i = 0; i < msize; i++)
    {
        for (int j = 0; j < msize; j++)
        {
            x.v[i][j] = in_0[i][j];
        }
    }

    puts("original matrix");
    matrix_show(&x);

    householder(&x, &R, &Q);

    puts("Q");
    matrix_show(&Q);
    puts("R");
    matrix_show(&R);

    //check orthoganal
    bool ifupper = matrix_is_upper_triangular(&R,0.00001);
    assert(ifupper==true);

    // to show their product is the input matrix
    mat_t m = matrix_mul(&Q, &R);
    puts("Q * R");
    matrix_show(&m);
    
    assert(equal_matrix(&m,&x)==1);
}

void test_invert_3by3matrix()
{
    int dim = 3;
    mat_t x;
    mat_t x_inv;
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x.v[i][j] = in_2[i][j];
        }
    }

    bool inv_ok = invert3by3matrix(&x, &x_inv);

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

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
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



    assert(inv_ok == true);
}

void eigen_vectors_3by3()
{

    printf("---test eigen_vectors_3by3\n");
    int dim = 3;
    mat_t x;
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x.v[i][j] = in_3[i][j];
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

    double result[3]={0};
    eigen_solve_eigenvalues(&x, 0.0001, 20, result);
    for (int i = 0; i < 3; i++)
    {
        printf(" %f", result[i]);
    }
    printf("\n");
    assert(equal_double(result[0], 0.0) == 1);
    assert(equal_double(result[1], 200.0) == 1);
    assert(equal_double(result[2], 0.0) == 1);


    mat_t eigen_vectors = eigen_solve_eigen_vectors(&x, result, 3, 3, 20);
    matrix_show(&eigen_vectors);
    assert(equal_double(eigen_vectors.v[0][1], 0.0) == 1);
    assert(equal_double(eigen_vectors.v[1][1], 0.3162) == 1);
    assert(equal_double(eigen_vectors.v[2][1], 0.9486) == 1);

}

void test_eigen_vectors_decomposition()
{
    printf("---test_eigen_vectors_decomposition\n");

    int dim = 3;
    mat_t x;
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x.v[i][j] = in_3[i][j];
        }
    }

    mat_t A = eigen_vector_decomposition(&x);

    mat_t A_trans =A;
    matrix_transpose(&A_trans);

    mat_t rst = matrix_mul(&A, &A_trans);

    puts("A");
    matrix_show(&A);
    puts("rst");
    matrix_show(&rst);

    assert(equal_matrix(&rst, &x) == 1);

}

void test_basic_operations(){

    printf("---test_basic_operations");

    mat_t a;
    vec_t u;
    vec_t m;

    for(int i=0;i<3;i++){
        for (int j=0;j<3;j++){
            a.v[i][j]=i+1;
        }
        u.v[i]=1;
        m.v[i]=i+1;
    }

    vec_t auv = matrix_mul_vec_add_vec(&a,&u,&m);
    vec_show(&auv);

    for(int i=0;i<3;i++){
        assert(auv.v[i]==(i+1)*4);
    }
}

int main()
{
    test_basic_operations();
    basic_qr_test();
    eigen_values_3by3();
    test_invert_3by3matrix();
    eigen_vectors_3by3();
    test_eigen_vectors_decomposition();
    return 0;
}