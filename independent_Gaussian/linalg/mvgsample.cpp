

/*
mean
0.8,4.4,2.6,6.2
cov
0,0 0.2
0,1 0.6
0,2 0.4
0,3 0.8
1,1 1.8
1,2 1.2
1,3 2.4
2,2 0.8
2,3 1.6
3,3 3.2

transform
0,0 0
0,1 0
0,2 -2.92872e-09
0,3 0.447214
1,0 -0
1,1 0
1,2 -2.76672e-18
1,3 1.34164
2,0 0
2,1 -0
2,2 2.92872e-10
2,3 0.894427
3,0 0
3,1 0
3,2 5.85744e-10
3,3 1.78885

iso
1.5
result
0.95
*/

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <math.h>
#include <stdio.h>

typedef struct
{
    // width of the matrix represented
    unsigned int n_col;
    // height of the matrix represented
    unsigned int n_row;
    // Pointer to the first element of the matrix represented
    double *data = NULL;
} Matrix;

typedef struct
{
    // width of the matrix represented
    unsigned int n_len;
    // Pointer to the first element of the matrix represented
    double *data = NULL;
} Vector;

#define MATRIX_IDX(M, r, c) (((r) * (M.n_col)) + (c))
#define MATRIX_IDX_INTO(M, r, c) (M.data[MATRIX_IDX(M, r, c)])

void print_matrix(Matrix m)
{
    printf("shape %d %d\n", m.n_col, m.n_row);
    for (uint i = 0; i < m.n_col; i++)
    {
        for (uint j = 0; j < m.n_row; j++)
        {
            printf("%f ", m.data[m.n_col * i + j]);
        }
        printf("\n");
    }
}

Matrix matrix_new(uint row, uint col)
{
    Matrix M;
    M.n_col = col;
    M.n_row = row;
    unsigned int size = M.n_row * M.n_col;
    M.data = (double *)malloc(size * sizeof(double));
    return M;
}

Matrix matrix_new_zero(uint row, uint col)
{
    Matrix M;
    M.n_col = col;
    M.n_row = row;
    unsigned int size = M.n_row * M.n_col;
    M.data = (double *)malloc(size * sizeof(double));
    for (int i = 0; i < M.n_row; i++)
    {
        for (int j = 0; j < M.n_col; j++)
        {
            MATRIX_IDX_INTO(M, i, j) = 0;
        }
    }
    return M;
}

// matrix multiplication
void matrix_multiply_into(Matrix reciever,
                          Matrix Mleft, Matrix Mright)
{
    assert(Mleft.n_col == Mright.n_row);
    // Zero out the reciever matrix.
    for (int i = 0; i < reciever.n_row; i++)
    {
        for (int j = 0; j < reciever.n_col; j++)
        {
            MATRIX_IDX_INTO(reciever, i, j) = 0;
        }
    }
    // Now multiply.
    for (int i = 0; i < Mleft.n_row; i++)
    {
        for (int k = 0; k < Mleft.n_col; k++)
        {
            for (int j = 0; j < Mright.n_col; j++)
            {
                MATRIX_IDX_INTO(reciever, i, j) +=
                    MATRIX_IDX_INTO(Mleft, i, k) * MATRIX_IDX_INTO(Mright, k, j);
            }
        }
    }
}

double vector_dot_product(Vector v1, Vector v2)
{
    assert(v1.n_len == v2.n_len);
    double dp = 0;
    for (int i = 0; i < v1.n_len; i++)
    {
        dp += v1.data[i] * v2.data[i];
    }
    return dp;
}

Vector vector_new(uint length)
{
    Vector v;
    v.n_len = length;
    unsigned int size = length;
    v.data = (double *)malloc(size * sizeof(double));
    return v;
}

Vector vector_normalize(Vector v)
{
    Vector vnorm = vector_new(v.n_len);
    // compute normal
    double norm_squared = vector_dot_product(v, v);
    double norm = sqrt(norm_squared);

    assert(norm != 0);
    for (int i = 0; i < v.n_len; i++)
    {
        vnorm.data[i] = v.data[i] / norm;
    }
    return vnorm;
}

double vector_norm_value (Vector v){
    double norm_squared = vector_dot_product(v, v);
    return sqrt(norm_squared);
}

void vector_normalize_self(Vector v)
{
    // compute normal value
    double norm = vector_norm_value(v);
    assert(norm != 0);

    for (int i = 0; i < v.n_len; i++)
    {
        v.data[i] = v.data[i] / norm;
    }
}



/* Create a new vector which contains a copy of a row of a matrix. */
Vector matrix_column_copy(Matrix M, int col)
{
    assert(0 <= col && col <= M.n_col - 1);
    Vector v = vector_new(M.n_row);
    for (uint i = 0; i < M.n_row; i++)
    {
        v.data[i] = MATRIX_IDX_INTO(M, i, col);
    }
    return v;
}

void vector_scalar_multiply_into(Vector reciever, Vector v, double s) {
    for(uint i = 0; i < v.n_len; i++) {
        reciever.data[i] = reciever.data[i] * s;
    }
}

void vector_free(Vector v){
    if(v.data!=NULL){
        free(v.data);
    }
}

void vector_subtract_into(Vector reciever, Vector v1, Vector v2) {
    assert(v1.n_len==v2.n_len);
    for(uint i = 0; i < v1.n_len; i++) {
        reciever.data[i] = v1.data[i] - v2.data[i];
    }
}

/* Copy the data in a vector into a column of a matrix.

   Note that this method modifies the matrix in place, it does not create a
   new matrix.
*/
void matrix_copy_vector_into_column(Matrix M, Vector v, int col) {
    assert(M.n_row == v.n_len);
    for(int i = 0; i < v.n_len; i++) {
        MATRIX_IDX_INTO(M, i, col) = v.data[i];
    }
}

// QR decomposition
/* Compute the QR decomposition of a matrix M.

   The current algorithm here is grahm-schmidt.  The columns of M are converted
   into an orthogonal basis by projection onto the previous vectors in the
   basis and subtracting out the projections from the current vector, resulting
   in a matrix Q.  These projections are accumulated into the matrix R.

   The resulting matricies Q and R satisfy:
     - M = Q * R
     - transpose(Q) * Q = Identity
     - R is upper triangular

   This decomposition gives a convienient way to solve general linear equations.
*/
// TODO: Split matrix decompositions out into their own module (matrix_decomp.c).
void matrix_qr_decomposition(Matrix M, Matrix rstq, Matrix rstr)
{

    // refer to https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/GramSchmidt.pdf
    //Matrix q = matrix_new_zero(M.n_row, M.n_col);
    //Matrix r = matrix_new(M.n_row, M.n_col);

    // struct qr_decomp *qr = qr_decomp_new(M);
    // struct matrix *q = matrix_new(M->n_row, M->n_col);
    // struct matrix *r = matrix_zeros(M->n_col, M->n_col);
    /* current_column:
       Initialized to the columns in M, in an outer loop.
       Transformed by subtracting out the projections of current_column onto
       the currently existing columns of Q.  After all projections are removed,
       what results is a vector orthogonal to all previous columns in Q.
    */
    // struct vector *current_column;
    /* current_unit_vector:
       Used to hold previously computed columns in Q, and the projections of
       columns of M onto them.
    */
    double current_dot_product;
    double norm;

    // go through each colum of the matrix
    for (int i = 0; i < M.n_col; i++)
    {
        Vector ai = matrix_column_copy(M, i);
        //compute ui based on ai
        for (int j = 0; j < i; j++)
        {
            // TODO: Allocate current_unit_vector one time, and then copy *into* this
            // vector, saving a malloc and free.
            Vector current_unit_vector = matrix_column_copy(rstq, j);

            current_dot_product = vector_dot_product(current_unit_vector, current_column);
            vector_scalar_multiply_into(current_unit_vector,
                                         current_unit_vector, current_dot_product);
            vector_subtract_into(current_column,
                                  current_column, current_unit_vector);
            vector_free(current_unit_vector);
            MATRIX_IDX_INTO(rstr, j, i) = current_dot_product;
        }
        norm = vector_norm_value(ai);
        // TODO: Check for zero norm here, indicating the the matrix is not full rank.
        // update r matrix
        MATRIX_IDX_INTO(rstr, i, i) = norm;
        vector_normalize_self(current_column);
        //update q matrix
        matrix_copy_vector_into_column(rstq, current_column, i);
        vector_free(current_column);
    }
}

int main()
{

    Matrix meanM = matrix_new(2, 2);

    // set the mean
    meanM.data[0] = 0.8;
    meanM.data[1] = 4.4;
    meanM.data[2] = 2.6;
    meanM.data[3] = 6.2;

    // set the cov
    double inputArray[] = {0.2, 0.6, 0.4, 0.8, 1.8, 1.2, 2.4, 0.8, 1.6, 3.2};
    Matrix covM = matrix_new(4, 4);

    uint index = 0;
    for (int i = 0; i < 4; i++)
    {
        for (int j = i; j < 4; j++)
        {
            MATRIX_IDX_INTO(covM, i, j) = inputArray[index];
            if (i != j)
            {
                MATRIX_IDX_INTO(covM, j, i) = inputArray[index];
            }
            index++;
        }
    }

    print_matrix(meanM);

    print_matrix(covM);

    // multivariant gaussian based on c
    // cholesky decomposition, todo
    // eigenvector decomposition
    // compute eigen vectors
    // using the qr decomposition
    // create q r matrix
    Matrix q = matrix_new_zero(4, 4);
    Matrix r = matrix_new_zero(4, 4);

    matrix_qr_decomposition(covM, q, r);

    printf("q:\n");
    print_matrix(q);

    printf("r:\n");
    print_matrix(r);


    // sample normal

    // Ly+u

    // TODO free matrix
}